"""Construction/edit policies and GraphAttnNet used by active model profiles.

Policies consume Observation and ActionMask through act/evaluate. They do not
run the legacy State-mutating route-generation loop. Additional upstream
architectures are imported explicitly from policy.models_additional.
"""
import copy
import logging as log

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, BatchNorm, GraphNorm

from ..city.graph import DEMAND_KEY, ROUTE_KEY
from ..common.tensors import get_update_at_mask, get_variable_slice_mask
from ..env.actions import ActionSpec, RouteAction
from ..env.const import RouteActionKind
from ..env.features import (
    Observation, aggr_edges_over_sequences, dims as feature_dims,
    extend_features, finite_time_features, global_state_features, halt_features, pair_features,
    trim_candidate_features,
)
from ..env.masks import (
    TORCH_FMIN, TORCH_FMAX, construction_halt_scores, edit_halt_scores,
    extension_mask, mask_action_family, path_mask, set_invalid_scores_to_fmin,
    trim_coverage_mask,
)
from ..env.state import RouteGenBatchState
from .const import (
    ATTN_DEFAULT_DROPOUT, DEFAULT_NONLIN, DEFAULT_PREFERENCE_GOALS,
    FEAT_NORM_MOMENTUM, MLP_DEFAULT_DROPOUT,
)
from .output import PolicyOut


class FeatureNorm(nn.Module):
    def __init__(self, dim, momentum=FEAT_NORM_MOMENTUM, unbiased=False):
        super().__init__()
        self.unbiased = unbiased
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.zeros(dim))
        self.register_buffer('avg_count', torch.tensor(1))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('frozen', torch.tensor(False))
        self.new_mean = None
        self.new_var = None
        self.new_count = 0

    def freeze(self):
        self.frozen[...] = True

    def extra_repr(self) -> str:
        return f'mean={self.running_mean}, var={self.running_var}, ' \
            f'initialized={self.initialized}, count={self.avg_count}'

    def forward(self, xx):
        assert xx.shape[-1] == self.running_mean.shape[-1], \
            "input tensor has the wrong dimensionality!"

        x_size = xx.shape[0]
        if x_size == 0:
            # the tensor is empty, so we don't need to do anything.
            return xx

        if not self.initialized or not self.frozen:
            # just use x's own mean and variance
            old_shape = xx.shape
            xx = xx.reshape(-1, xx.shape[-1])
            x_mean = xx.mean(0).detach()
            # Legacy profiles use population variance; Holliday uses sample
            # variance. A singleton always uses the finite population fallback.
            x_var = xx.var(0, unbiased=self.unbiased and xx.shape[0] > 1).detach()
            # we're done with the need for flattened x, so reshape it
            xx = xx.reshape(old_shape)

        if not self.frozen:
            if self.new_mean is None:
                # initialize the new-samples mean
                self.new_mean = x_mean
                self.new_var = x_var
                self.new_count = x_size
            else:
                # update new-samples mean
                nmrtr = (self.new_mean * self.new_count + x_mean * x_size)
                updated_count = self.new_count + x_size
                updated_mean = nmrtr / updated_count

                # update new-samples variance using the variance-combining formula:
                # v_c = n_1(v_1 + (m_1 - m_c)^2) + n_2 * (v_2 + (m_1 - m_2)^2)
                #        / (n_1 + n_2)
                mean_diff_1 = (self.new_mean - updated_mean) ** 2
                mean_diff_2 = (x_mean - updated_mean) ** 2
                prev_part = self.new_count * (self.new_var + mean_diff_1)
                sample_part = x_size * (x_var + mean_diff_2)
                self.new_var = (prev_part + sample_part) / updated_count

                # update count of new samples
                self.new_count = updated_count

        if not self.initialized:
            # we don't have any running statistics yet, so just normalize by
             # minibatch statistics.
            shift = x_mean
            denom = x_var.sqrt()

        else:
            # avoid division by 0
            shift = self.running_mean
            denom = self.running_var.sqrt()

        # avoid division by nan or 0. nans will occur in variance if batch
         # size is 1.
        denom[(denom == 0) | denom.isnan()] = 1
        out = (xx - shift) / denom
        return out

    def update(self):
        if self.frozen:
            # do nothing
            return

        if self.new_mean is not None:
            if not self.initialized:
                # just set the initial statistics
                self.running_mean[...] = self.new_mean
                self.running_var[...] = self.new_var
                self.avg_count = torch.tensor(self.new_count)
                self.initialized[...] = True
            else:
                # update running statistics
                # scale update size in proportion to how big the sample is
                alpha = self.momentum * self.new_count / self.avg_count
                # if new count is *much* bigger than old, don't let alpha
                # be greater than 1...that would make things wierd
                alpha = min(alpha, 1.0)
                updated_mean = alpha * self.new_mean + \
                    (1 - alpha) * self.running_mean

                # implements the variance-combing formula, assuming
                 # n_1 / (n_1 + n_2) = alpha, n_2 / (n_1 + n_2) = 1 - alpha
                t1 = self.new_var + (self.new_mean - updated_mean) ** 2
                t2 = self.running_var + (self.running_mean - updated_mean) ** 2
                self.running_var[...] = alpha * t1 + (1 - alpha) * t2

                # update the running mean *after* using it to compute the new
                 # running variance
                self.running_mean[...] = updated_mean
                self.avg_count = self.momentum * self.new_count + \
                    (1 - self.momentum) * self.avg_count

            self.new_mean = None
            self.new_var = None
            self.new_count = 0

        else:
            log.warning("FeatureNorm was updated without any samples!")


class RouteScorer(nn.Module):
    # Feature spec: order of the cost-weight slots inside global state
    # features, locked to the trained checkpoints (see DEFAULT_PREFERENCE_GOALS).
    preference_goals = DEFAULT_PREFERENCE_GOALS

    def __init__(self, embed_dim, nonlin_type, dropout, n_mlp_layers=2,
                 mlp_width=None, n_extra_feats=14, preference_goals=None):
        super().__init__()
        if preference_goals is not None:
            self.preference_goals = tuple(preference_goals)
        self.embed_dim = embed_dim
        # route_feats (2) + global_state_features. Default 14 = 2 + 12; the edit
        # model with adjustment conditioning passes 16 = 2 + 14.
        self.n_extra_feats = n_extra_feats
        self.dropout = dropout
        if mlp_width is None:
            mlp_width = self.in_dim * 2
        self.out_mlp = get_mlp(n_mlp_layers, mlp_width, nonlin_type,
                               dropout, in_dim=self.in_dim, out_dim=1)
        self.extras_norm = FeatureNorm(self.n_extra_feats)

    @property
    def in_dim(self):
        return self.embed_dim * 2 + self.n_extra_feats

    def forward(self, state, node_descs, route_idxs, route_time,
                node_padding_mask=None):
        # assemble route sequences
        route_gather_idxs = route_idxs * (route_idxs > -1)
        route_gather_idxs = \
            route_gather_idxs.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        route_seqs = node_descs.gather(1, route_gather_idxs)

        route_len = (route_idxs > -1).sum(dim=1, dtype=torch.float)
        extra_feats = halt_features(state, route_time, route_len)

        # pass sequences through encoder
        route_desc = self.encode(node_descs, route_seqs, node_padding_mask,
                                 route_idxs == -1)

        global_node_desc = node_descs.mean(dim=1)

        # normalize extra features
        extra_feats = self.extras_norm(extra_feats)
        # pass encoding and features through MLP
        in_vec = torch.cat((route_desc,
                            global_node_desc,
                            extra_feats), dim=1)
        scores = self.out_mlp(in_vec)
        return scores

    def encode(self, node_descs, route_seqs, node_pad_mask=None,
               route_pad_mask=None, embed_pos=False):
        raise NotImplementedError


class RouteEndpointsScorer(RouteScorer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        placeholder = torch.zeros((1, 1, self.embed_dim * 2),
                                  dtype=torch.float32)
        self.register_buffer('placeholder', placeholder)

    @property
    def in_dim(self):
        return self.n_extra_feats + self.embed_dim * 3

    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask):
        # get first and last node in each route sequence
        first_nodes = route_seqs[:, 0]
        idxs = (~route_pad_mask).sum(dim=1, keepdim=True) - 1
        route_is_empty = route_pad_mask[:, 0]
        # negative indices will occur if the route is empty
        idxs[route_is_empty] = 0
        idxs = idxs[..., None].expand(-1, -1, route_seqs.shape[-1])
        last_nodes = route_seqs.gather(1, idxs).squeeze(1)
        descs = torch.cat((first_nodes, last_nodes), dim=-1)
        # where the route is empty, use a placeholder
        descs[route_is_empty] = self.placeholder
        # append the mean descriptor of the nodes
        # mean_descs = mean_pool_sequence(node_descs, node_pad_mask)
        # descs = torch.cat((descs, mean_descs), dim=-1)
        return descs



class RouteGeneratorBase(nn.Module):
    # Feature spec: order of the cost-weight slots in every positional
    # feature vector this model consumes (global state features, edge
    # planes, decoder context). Locked to the trained checkpoints (see
    # DEFAULT_PREFERENCE_GOALS); a model trained with a different order overrides
    # this next to its checkpoint.
    preference_goals = DEFAULT_PREFERENCE_GOALS

    def __init__(self, backbone_net, mean_stop_time_s,
                 embed_dim, n_nodepair_layers, nonlin_type=DEFAULT_NONLIN,
                 dropout=MLP_DEFAULT_DROPOUT, symmetric_routes=True,
                 temperature=1, fixed_frequency_Hz=0.00074,
                 only_routes_with_demand_are_valid=False,
                 low_memory_mode=False, feature_set=None):
        """nonlin_type: a nonlin, or the string name of a nonlin, like 'ReLU'
            or 'LeakyReLU' or 'Tanh'"""
        super().__init__()
        if feature_set is None:
            raise ValueError("policy requires a feature set")
        self.feature_set = feature_set
        self.preference_goals = feature_set.preference_goals
        # make this a parameter of the model so it will be kept in state_dict
        # mst = torch.tensor(mean_stop_time_s, dtype=torch.float32)
        self.mean_stop_time_s = mean_stop_time_s

        self.dropout = dropout
        if isinstance(nonlin_type, str):
            self.nonlin_type = getattr(nn, nonlin_type)
        else:
            self.nonlin_type = nonlin_type
        self.embed_dim = embed_dim
        self.n_nodepair_layers = n_nodepair_layers
        self.backbone_net = backbone_net
        self.symmetric_routes = symmetric_routes
        self.temperature = temperature
        self.fixed_freq = fixed_frequency_Hz
        self.only_routes_with_demand_are_valid = \
            only_routes_with_demand_are_valid
        self.low_memory_mode = low_memory_mode

        self.edge_feat_dim = backbone_net.in_edge_dim
        if self.edge_feat_dim is None:
            self.edge_feat_dim = 0
        # + 2 for the cost weights, + 2 for the two n_routes features
        if symmetric_routes:
            self.full_nodepair_dim = self.edge_feat_dim + embed_dim
        else:
            self.full_nodepair_dim = self.edge_feat_dim + embed_dim * 2

        # edge feats: demand, drive time, has route, route time, has street,
         # street time, has 1-transfer path, has 2-transfer path, has any path,
         # shortest transit path time, is same node
        if self.edge_feat_dim > 0:
            self.edge_norm = FeatureNorm(self.edge_feat_dim - 1)
            self.time_norm = FeatureNorm(1)
        self.node_norm = FeatureNorm(backbone_net.in_node_dim)

    @property
    def edge_key_order(self):
        return (DEMAND_KEY, ROUTE_KEY)

    def update_and_freeze_feature_norms(self):
        self.update_feature_norms()
        self.freeze_feature_norms()

    def update_feature_norms(self):
        for mod in self.modules():
            if isinstance(mod, FeatureNorm):
                mod.update()

    def freeze_feature_norms(self):
        for mod in self.modules():
            if isinstance(mod, FeatureNorm):
                mod.freeze()



    def _get_edge_features(self, state):
        raw = pair_features(state, use_redundancy=self.use_redundancy_features,
                            preference_goals=self.preference_goals)
        edge_features = self.edge_norm(raw)
        finite_drive_times = finite_time_features(state, state.drive_times)
        drive_times = self.time_norm(finite_drive_times[..., None])
        return torch.cat((edge_features, drive_times), dim=-1)

    @staticmethod
    def fold_node_descs(node_descs, state):
        # fold the node description vectors so that the batch dimension
         # is separate from the nodes-in-a-batch-elem dimension
        folded_node_descs = torch.zeros((state.batch_size, state.max_n_nodes,
                                         node_descs.shape[-1]),
                                         device=node_descs.device)
        node_pad_mask = torch.zeros((state.batch_size, state.max_n_nodes),
                                    dtype=bool, device=node_descs.device)
        node_offset = 0
        for bi, num_nodes in enumerate(state.n_nodes):
            num_nodes = int(num_nodes.item())
            next_offset = node_offset + num_nodes
            folded_node_descs[bi, :num_nodes, :node_descs.shape[-1]] = \
                node_descs[node_offset:next_offset]
            node_pad_mask[bi, num_nodes:] = True
            node_offset = next_offset
        if node_offset != node_descs.shape[0]:
            raise ValueError(
                "Node embedding count does not match batched graph nodes: "
                f"{node_offset} vs {node_descs.shape[0]}"
            )

        return folded_node_descs, node_pad_mask


class PathCombiningRouteGenerator(RouteGeneratorBase):
    action_spec = ActionSpec("construction")
    def __init__(self, *args, n_pathscorer_layers=3, pathscorer_hidden_dim=16,
                 halt_scorer_type='endpoints', n_halt_layers=3, n_halt_heads=4,
                 force_linking_unlinked=False, logit_clip=None,
                 serial_halting=True, max_act_len=None,
                 use_redundancy_features=False, feature_norm_unbiased=False,
                 halt_scorer: RouteScorer | None = None, **kwargs):
        """Generates routes by combining shortest paths.

        n_pathscorer_layers -- number of layers in the MLP that updates path
            scores
        pathscorer_hidden_dim -- number of hidden units in the MLP that updates
            path scores
        halt_scorer_type -- 'endpoints' for the built-in scorer.
        halt_scorer -- an optional, explicitly constructed RouteScorer.
            It receives the same node embeddings and configured features.
        n_halt_layers -- number of layers in the network that computes the
            halting score.
        n_halt_heads -- number of heads in the network that computes the
            halting score, if it has heads.
        force_linking_unlinked -- if True, then the model is forced to extend
            routes in ways that link unlinked node pairs.
        logit_clip -- if not None, then the logits of path extensions are
            clipped to this value.
        serial_halting -- if True, the halt score is used to make a binary
            halt-or-continue choice before choosing each extension.  If False,
            the 'halt' action is part of the same set of actions as the
            extensions.
        """
        super().__init__(*args, **kwargs)
        self.logit_clip = logit_clip
        self.serial_halting = serial_halting
        self.use_redundancy_features = use_redundancy_features
        self.n_redundancy_global_feats = 5 if use_redundancy_features else 0
        self.n_redundancy_path_feats = 5 if use_redundancy_features else 0
        if use_redundancy_features and self.edge_feat_dim != 18:
            raise ValueError(
                "Redundancy-aware path combining expects a backbone with "
                f"in_edge_dim=18, got {self.edge_feat_dim}")
        # only paths shorter than this can be added to a route
        self.max_act_len = max_act_len
        self.nodepair_scorer = get_mlp(self.n_nodepair_layers,
                                       self.embed_dim,
                                       self.nonlin_type, self.dropout,
                                       in_dim=self.full_nodepair_dim,
                                       out_dim=1)

        assert self.only_routes_with_demand_are_valid is False, 'not supported'
        self.force_linking_unlinked = force_linking_unlinked
        self.route_step_counts_log = []

        # Gated adjustment conditioning: when > 0, get_global_state_features
        # appends this many extra features (target + normalized weight), so the
        # scorers that consume global features must widen by the same amount.
        # Default 0 keeps the construction-model dims (and frozen weights) intact.
        self.n_adjustment_cond_feats = len(self.feature_set.adjustment_features)
        _halt_n_extra = feature_dims(self.feature_set)["halt"]

        path_scorer_indim = feature_dims(self.feature_set)["extend"]
        # self.path_input_norm = FeatureNorm(FEAT_NORM_MOMENTUM,
        #                                    path_scorer_indim - 1)
        self.path_scorer = nn.Sequential(
            FeatureNorm(path_scorer_indim),
            get_mlp(n_pathscorer_layers, pathscorer_hidden_dim,
                    self.nonlin_type, self.dropout, in_dim=path_scorer_indim,
                    out_dim=1)
        )
        if halt_scorer is not None:
            if not isinstance(halt_scorer, RouteScorer):
                raise TypeError("halt_scorer must implement RouteScorer")
            self.halt_scorer = halt_scorer
        elif halt_scorer_type == "endpoints":
            self.halt_scorer = RouteEndpointsScorer(self.embed_dim,
                                                    self.nonlin_type,
                                                    self.dropout,
                                                    n_halt_layers,
                                                    n_extra_feats=_halt_n_extra,
                                                    preference_goals=self.preference_goals)
        else:
            raise ValueError(
                "The default halt scorer is 'endpoints'; pass an explicit "
                "halt_scorer from policy.models_additional for another scorer"
            )

        # The upstream construction profile uses sample variance. Preserve
        # legacy checkpoint profiles' population variance by default.
        for module in self.modules():
            if isinstance(module, FeatureNorm):
                module.unbiased = feature_norm_unbiased

        # self.value_net = ValueNetScorer(self.embed_dim, self.nonlin_type,
        #                                 self.dropout, n_halt_layers)

    def clear_step_counts_log(self):
        """Clear the route step counts accumulated by rollout."""
        self.route_step_counts_log = []

    def _log_step_counts(self, counts: list):
        self.route_step_counts_log.extend(counts)

    def act(self, obs, mask, greedy=False):
        if not isinstance(obs, Observation):
            raise TypeError("policy execution requires an Observation")
        return self._action_output(obs, mask, greedy, None)

    def forward(self, obs, mask, greedy=False):
        return self.act(obs, mask, greedy)

    def evaluate(self, obs, mask, action):
        if not isinstance(obs, Observation):
            raise TypeError("policy execution requires an Observation")
        return self._action_output(obs, mask, False, action)

    def _action_output(self, obs, mask, greedy, action):
        indices, logp, entropy, log_probs = self._decode(
            obs, greedy, None if action is None else action.indices,
            obs.encoding, mask=mask)
        kind = torch.where(indices[:, 0] < 0, RouteActionKind.HALT, RouteActionKind.EXTEND)
        return PolicyOut(RouteAction(kind, indices), logp, entropy, log_probs.exp())


    def _decode(self, state, greedy=False, actions=None,
             precalc_data=None, *, mask):
        """Take an action for the given state.
        actions -- a batch_size x 2 tensor of predetermined actions to take. If
            None, the actions will be chosen by the model.  If the value is
            (-1, -1), that means to halt.  Otherwise, the first and second
            columns give the starting and ending nodes of the path segment to
            add.
        """
        log.debug("stepping")
        if not isinstance(state, Observation):
            raise TypeError("policy execution requires an Observation")

        if precalc_data is None:
            node_descs, node_pad_mask, np_embeds, path_scores = \
                self._encode_graph(state)
        else:
            node_descs, node_pad_mask, np_embeds, path_scores = precalc_data

        path_seqs = state.get_shortest_path_sequences()
        batch_size = state.batch_size

        # don't choose segments greater than the max route length
        path_lens, current_route_lens, paths_are_invalid = path_mask(
            state, path_seqs, self.max_act_len, self.force_linking_unlinked)
        path_scores = set_invalid_scores_to_fmin(path_scores,
                                                      ~paths_are_invalid)
        no_valid_options = paths_are_invalid.all(-1).all(-1)
        old_route_is_done = state.is_done() | no_valid_options

        # determine if we're extending a route or starting a new one
        # for starting, we ignore the halt choice
        starting = ~state.is_done() & (current_route_lens == 0)

        # update scores and get validity for extensions to non-empty routes
        padding = (0,1, 0,1, 0,0)
        drive_times = torch.nn.functional.pad(state.drive_times, padding)
        padding = (0, 0) + padding
        np_embeds = torch.nn.functional.pad(np_embeds, padding)
        are_on_routes = torch.zeros((batch_size, state.max_n_nodes + 1),
                                    device=state.device).bool()
        batch_idxs = state.batch_indices
        are_on_routes[batch_idxs[:, None], state.current_routes] = True
        # make padding column False
        are_on_routes[:, -1] = False
        getext_args = (state, current_route_lens, are_on_routes, path_seqs,
                       np_embeds, drive_times, path_scores, path_lens,
                       paths_are_invalid)
        prev_scores, prev_valid = self._get_extension_scores(*getext_args,
            before_or_after='before')
        next_scores, next_valid = self._get_extension_scores(*getext_args,
            before_or_after='after')

        # compute the scores for halting
        halt_scores = self.halt_scorer(state, node_descs, state.current_routes,
                                       state.current_route_time, node_pad_mask)
        ext_valid = prev_valid | next_valid
        halt_scores = construction_halt_scores(
            state, halt_scores, current_route_lens, old_route_is_done,
            starting, ext_valid, self.force_linking_unlinked, mask.halt)

        ext_scores = prev_scores * prev_valid + next_scores * next_valid + \
            TORCH_FMIN * ~ext_valid
        start_scores = self._update_path_scores(state, path_scores, path_lens,
                                                state.drive_times,
                                                candidate_paths=path_seqs)
        # The conditional continue branch must not depend on the replayed
        # action: a sampled halt still needs every extension in its KL.
        continuing = (~starting & ~old_route_is_done)[:, None, None]
        all_scores = torch.where(continuing, ext_scores, start_scores)
        all_valid = torch.where(continuing, ext_valid, ~paths_are_invalid)
        action_log_probs = _joint_route_action_log_probs(
            all_scores.flatten(1), all_valid.flatten(1), halt_scores,
            self.serial_halting, self.temperature, old_route_is_done)

        route_is_done = old_route_is_done
        if self.serial_halting:
            # decide whether to halt
            continue_scores = -halt_scores
            cont_or_halt = torch.cat((continue_scores, halt_scores), dim=-1)

            given_halt_actions = None
            if actions is not None:
                # predetermined actions were given, so force them to be chosen
                given_halt_actions = (actions[:, 0] == -1).to(torch.long)

            halt, corh_logit, corh_ent = select(cont_or_halt, not greedy,
                                                self.temperature,
                                                selection=given_halt_actions)
            assert (corh_logit > TORCH_FMIN).all(), "halt score is too low!"
            corh_logit = corh_logit.squeeze(1)
            chose_halt = halt.squeeze(1).bool()

            # update doneness
            route_is_done = old_route_is_done | chose_halt

        # for extending, we care whether we chose to halt
        extending = ~starting & ~route_is_done
        xtnding_exp = extending[:, None, None]
        ppe_scores = start_scores * ~xtnding_exp + ext_scores * xtnding_exp

        # select an option
        flat_path_scores = ppe_scores.reshape(batch_size, -1)
        if not self.serial_halting:
            flat_path_scores = torch.cat((flat_path_scores, halt_scores),
                                         dim=-1)
            halt_idx = flat_path_scores.shape[-1] - 1

        given_ext_actions = None
        if actions is not None:
            # predetermined actions were given, so force them to be chosen
            given_ext_actions = actions[:, 0] * state.n_nodes + actions[:, 1]
            if self.serial_halting:
                given_ext_actions[route_is_done] = 0
            else:
                given_ext_actions[route_is_done] = halt_idx
                given_ext_actions[actions[:, 0] < 0] = halt_idx

        flat_idxs, ext_logit, ext_ent = select(flat_path_scores, not greedy,
                                               self.temperature,
                                               selection=given_ext_actions)

        flat_idxs = flat_idxs.squeeze(-1)
        ext_logit = ext_logit.squeeze(-1)
        if self.serial_halting:
            # these don't contribute if we chose to halt
            ext_logit[chose_halt] = 0.0
            ext_ent[chose_halt] = 0.0
        else:
            # mark the halt action as chosen
            chose_halt = flat_idxs == halt_idx
            route_is_done = route_is_done | chose_halt

        assert ((ext_logit > TORCH_FMIN) | route_is_done).all()

        from_idxs = torch.div(flat_idxs, state.n_nodes, rounding_mode='floor')
        to_idxs = flat_idxs % state.n_nodes
        folded_idxs = torch.stack((from_idxs, to_idxs), dim=-1)
        set_to_minus1 = route_is_done[:, None]
        folded_idxs = folded_idxs * ~set_to_minus1 + -1 * set_to_minus1

        # combine the logits and entropies of the decisions
        if self.serial_halting:
            logits = corh_logit + ext_logit
            entropy = corh_ent + ext_ent
        else:
            logits = ext_logit
            entropy = ext_ent

        # is_halt = (folded_idxs == -1).any(-1)
        # badlen = (current_route_lens < state.min_route_len) & \
        #     (current_route_lens > 0)
        # if (is_halt & badlen).any():
        #     log.warning("Halting a route that's too short!")

        return folded_idxs, logits, entropy, action_log_probs

    def _encode_graph(self, state: RouteGenBatchState):
        if not isinstance(state, Observation):
            raise TypeError("policy execution requires an Observation")
        # assemble route data objects
        input_batch = copy.copy(state.graph_data)
        input_batch.x = state.norm_node_features
        input_batch.edge_index = input_batch[DEMAND_KEY].edge_index
        edge_feats_square = self._get_edge_features(state)
        if isinstance(self.backbone_net, GraphAttnNet) or self.backbone_net.gives_edge_features:
            isedge_mask = torch.eye(state.max_n_nodes, device=state.device)
            isedge_mask = ~(isedge_mask.bool())
            if state.max_n_nodes * state.batch_size > input_batch.num_nodes:
                # some graphs are smaller than the max size, so mask out the
                 # invalid node locations.
                isedge_mask = isedge_mask.repeat(state.batch_size, 1, 1)
                for bi in range(state.batch_size):
                    isedge_mask[bi, state.n_nodes[bi]:] = False
                    isedge_mask[bi, :, :state.n_nodes[bi]:] = False
            else:
                isedge_mask = isedge_mask.expand(state.batch_size, -1, -1)

            edge_feats_flat = edge_feats_square[isedge_mask]
            input_batch.edge_attr = edge_feats_flat

        path_seqs = state.get_shortest_path_sequences()

        # run GNN forward
        if self.backbone_net.gives_edge_features:
            node_descs, _ = self.backbone_net(input_batch)
        else:
            node_descs = self.backbone_net(input_batch)

        node_descs, node_pad_mask = self.fold_node_descs(node_descs, state)
        if self.symmetric_routes:
            # just sum them
            np_embeds = node_descs[:, None] + node_descs[:, :, None]
        else:
            # concatenate them along the last dimension
            np_embeds = get_node_pair_descs(node_descs)
        np_embeds = torch.cat((np_embeds, edge_feats_square), dim=-1)
        np_scores = self.nodepair_scorer(np_embeds).squeeze(-1)
        score_clip = 10**6
        if (not torch.isfinite(np_scores).all()) or \
                (np_scores.abs() >= score_clip).any():
            log.warning(
                "Nodepair scores exceeded the safe range; clipping for "
                "route generation.")
            np_scores = torch.nan_to_num(
                np_scores, nan=0.0, posinf=score_clip, neginf=-score_clip)
            np_scores = np_scores.clamp(-score_clip, score_clip)

        path_scores = aggr_edges_over_sequences(path_seqs,
                                                   np_scores[..., None], 'sum')
        path_scores.squeeze_(-1)

        return node_descs, node_pad_mask, np_embeds, path_scores

    def get_critic_features(self, state: RouteGenBatchState):
        """Pooled GNN node embeddings for a shared-encoder critic.

        Returns ``[batch, 2 * embed_dim + n_global_features]``:
        graph mean-pool || current-route mean-pool || global state feats.

        The encoder forward is detached: value-function gradients do not
        flow back into the actor's GNN. This gives the critic the same
        rich state representation the actor sees, without destabilizing
        the policy via mixed value/policy gradients.
        """
        if not isinstance(state, Observation):
            raise TypeError("policy execution requires an Observation")
        with torch.no_grad():
            node_descs, node_pad_mask, _, _ = self._encode_graph(state)
        node_descs = node_descs.detach()

        # Graph-level mean pool over valid (non-padding) nodes.
        valid = (~node_pad_mask).to(node_descs.dtype).unsqueeze(-1)
        graph_embed = (node_descs * valid).sum(dim=1) / \
            valid.sum(dim=1).clamp_min(1.0)

        # Pool the embeddings of the nodes on the route currently being
        # edited so the critic sees the specific route, not just the graph.
        routes = state.current_routes
        route_mask = (routes >= 0)
        safe_routes = routes.clamp_min(0)
        gather_idx = safe_routes.unsqueeze(-1).expand(
            -1, -1, node_descs.shape[-1])
        route_node_descs = torch.gather(node_descs, 1, gather_idx)
        route_valid = route_mask.to(node_descs.dtype).unsqueeze(-1)
        route_embed = (route_node_descs * route_valid).sum(dim=1) / \
            route_valid.sum(dim=1).clamp_min(1.0)

        global_feats = global_state_features(state,
            preference_goals=self.preference_goals).to(node_descs.dtype)
        return torch.cat([graph_embed, route_embed, global_feats], dim=-1)

    def _get_extension_scores(self, state: RouteGenBatchState, route_lens,
                              are_on_routes, all_paths, nodepair_embeds,
                              drive_times, path_scores, path_lens,
                              invalid_terms, before_or_after='after'):
        log.debug("get extension scores")
        assert before_or_after in ['before', 'after']
        lens_wo_term = route_lens - 1
        # clamp for cases where the route is empty, so we don't get -1
        lens_wo_term.clamp_(min=0)
        times_from_start = state.current_route_times_from_start
        routes = state.current_routes
        final_times = times_from_start.gather(1, lens_wo_term[:, None])
        times_to_end = (times_from_start - final_times).abs()
        if before_or_after == 'after':
            # terminal is last node
            mask = get_variable_slice_mask(routes, 1, froms=lens_wo_term)
            routes_but_terminal = get_update_at_mask(routes, mask, -1)[:, :-1]
            terminal_nodes = routes.gather(1, lens_wo_term[:, None]).squeeze(-1)
            times_to_end = get_update_at_mask(times_to_end, mask, 0)[:, :-1]
        else:
            # terminal is first node
            routes_but_terminal = routes[:, 1:].clone()
            terminal_nodes = routes[:, 0]
            times_to_end = times_to_end[:, 1:]
            # swap from- and to-node axes so indexing the first node axis
             # by the terminal gives data on edges/paths *to* the terminal,
             # instead of *from* the terminal.
            drive_times = drive_times.transpose(-2, -1)
            nodepair_embeds = nodepair_embeds.transpose(-3, -2)
            path_scores = path_scores.transpose(-2, -1)
            path_lens = path_lens.transpose(-2, -1)
            all_paths = all_paths.transpose(-3, -2)

        # get possible extensions
        batch_idxs = state.batch_indices
        raw_extension_paths = all_paths[batch_idxs, terminal_nodes]
        extension_paths = raw_extension_paths.clone()
        # ignore the terminal itself in the extensions
        extension_paths[extension_paths == terminal_nodes[:, None, None]] = -1

        times_from_end = drive_times[batch_idxs, terminal_nodes]
        # batch_size x route_len - 1 x n_nodes + 1
        times_on_route = times_from_end[:, None] + times_to_end[:, :, None]

        # compute scores for all edges from each node on route
        embeds_on_route = nodepair_embeds[batch_idxs[:, None],
                                          routes_but_terminal]

        # use a neural network to compute scores and lengths
        times_on_route = finite_time_features(state, times_on_route)
        norm_times = self.time_norm(times_on_route.flatten(0, 2)[:, None])
        norm_times = norm_times.reshape_as(times_on_route)[..., None]
        log.debug("updating edge scores...")
        embeds_w_newtimes = torch.cat((embeds_on_route[..., :-1], norm_times),
                                      dim=-1)
        from_route_edge_scores = self.nodepair_scorer(embeds_w_newtimes)
        from_route_edge_scores = from_route_edge_scores.squeeze(-1)
        log.debug("edge scores updated")

        # zero out scores corresponding to dummy route nodes
        from_route_edge_scores = \
            from_route_edge_scores * (routes_but_terminal > -1)[..., None]
        # sum over nodes on route to get "score" of visiting each node before/
         # after the current route.
        from_route_node_scores = from_route_edge_scores.sum(dim=-2)
        # compute scores for each possible following sequence
        # ignore first node of each sequence, since it's the terminal
        from_route_pathnode_scores = \
            from_route_node_scores[batch_idxs[:, None, None], extension_paths]
        # zero locations corresponding to "dummy" extension path nodes
        from_route_pathnode_scores[extension_paths == -1] = 0
        from_route_path_scores = from_route_pathnode_scores.sum(dim=-1)

        # add score of route from end node to each node not on route, and add
         # current score
        innate_scores = path_scores[batch_idxs, terminal_nodes]
        ext_scores = from_route_path_scores + innate_scores

        # apply final transformer of lengths
        new_path_lens = path_lens[batch_idxs, terminal_nodes]
        # The shortest path contains the shared route endpoint.
        new_lens = new_path_lens + route_lens[:, None] - 1
        # max over the stops already on the route, and drop the dummy row
        added_drive_times = times_from_end[..., :-1]

        ext_scores = self._update_path_scores(state, ext_scores, new_path_lens,
                                              added_drive_times, route_lens,
                                              times_to_end.max(-1)[0],
                                              raw_extension_paths)

        assert ext_scores.isfinite().all()

        invalid = extension_mask(state, are_on_routes, extension_paths,
                                 raw_extension_paths, path_lens, terminal_nodes,
                                 new_lens, self.force_linking_unlinked)

        out_scores = torch.zeros_like(path_scores)
        out_scores[batch_idxs, terminal_nodes] = ext_scores
        out_valids = torch.zeros_like(path_scores, dtype=bool)
        out_valids[batch_idxs, terminal_nodes] = ~invalid

        if before_or_after == 'before':
            # swap from- and to-node axes back
            out_scores = out_scores.transpose(-2, -1)
            out_valids = out_valids.transpose(-2, -1)

        out_valids &= ~invalid_terms
        out_scores = get_update_at_mask(out_scores, ~out_valids, TORCH_FMIN)

        log.debug("extension scores gotten")

        return out_scores, out_valids

    # def _update_path_scores(self, state, base_path_scores, new_path_lens,
    #                         new_drive_times):
    #     """Update the path scores based on the new path lengths and drive times.
    #     """
    #     if (base_path_scores > 10**5).any():
    #         log.warning("base path scores are blowing up!")

    #     is_valid = base_path_scores > TORCH_FMIN
    #     update_in = torch.stack((base_path_scores, new_drive_times,
    #                              new_path_lens), dim=-1)

    #     global_feat = global_state_features(state, )
    #     update_in = nn.functional.pad(update_in, (0, global_feat.shape[-1]))
    #     for _ in range(update_in.ndim - global_feat.ndim):
    #         global_feat = global_feat.unsqueeze(1)
    #     update_in[..., -global_feat.shape[-1]:] = global_feat
    #     update_in = update_in[is_valid]

    #     updated_path_scores_wo_infs = self.path_scorer(update_in).squeeze(-1)
    #     if self.logit_clip is not None:
    #         upswi = torch.tanh(updated_path_scores_wo_infs)
    #         updated_path_scores_wo_infs = torch.clamp(upswi, -self.logit_clip,
    #                                                   self.logit_clip)
    #     updated_path_scores = torch.full_like(base_path_scores, TORCH_FMIN)
    #     updated_path_scores[is_valid] = updated_path_scores_wo_infs

    #     return updated_path_scores

    def _update_path_scores(self, state: RouteGenBatchState, base_path_scores,
                            new_path_lens, new_drive_times, prev_len=None,
                            prev_drive_time=None, candidate_paths=None):
        """Update the path scores based on the new path lengths and drive times.
        """
        is_valid = base_path_scores > TORCH_FMIN
        update_in = extend_features(state, base_path_scores, new_path_lens,
                                    new_drive_times, prev_len, prev_drive_time,
                                    candidate_paths)
        update_in = update_in[is_valid]

        updated_path_scores_wo_infs = self.path_scorer(update_in).squeeze(-1)
        if self.logit_clip is not None:
            upswi = torch.tanh(updated_path_scores_wo_infs)
            updated_path_scores_wo_infs = torch.clamp(upswi, -self.logit_clip,
                                                      self.logit_clip)
        updated_path_scores = torch.full_like(base_path_scores, TORCH_FMIN)
        updated_path_scores[is_valid] = updated_path_scores_wo_infs

        return updated_path_scores




class TrimPathCombiningRouteGenerator(PathCombiningRouteGenerator):
    """Path-combining generator with explicit route-trimming actions.

    The legacy generator can start/extend/halt routes.  This subclass keeps
    that behaviour but adds two extra action kinds for non-empty current routes:
    trim_start keeps the suffix beginning at a chosen existing stop, and
    trim_end keeps the prefix ending at a chosen existing stop.
    """

    supports_trim_actions = True
    action_spec = ActionSpec("edit")

    def _action_output(self, obs, mask, greedy, action):
        kind, indices, logp, entropy, log_probs = self._decode_actions(
            obs, greedy, None if action is None else action.indices,
            obs.encoding, None if action is None else action.kind,
            mask=mask)
        return PolicyOut(RouteAction(kind, indices), logp, entropy, log_probs.exp())


    def __init__(self, *args, n_trim_scorer_layers=3,
                 trim_scorer_hidden_dim=16,
                 forbid_halt_when_overlong=False,
                 allow_trim_below_min=False, **kwargs):
        super().__init__(*args, **kwargs)
        # When True, trim candidates may shorten a route below min_route_len
        # (down to an absolute floor of 2 stops). halt stays masked while the
        # route is below min, so the agent must extend back up before halting.
        self.allow_trim_below_min = allow_trim_below_min
        self.trim_base_feat_dim = 6
        self.trim_overlap_feat_dim = 8
        self.trim_redundancy_feat_dim = \
            5 if self.use_redundancy_features else 0
        # Matches global_features(RouteGenBatchState, ): 12 base,
        # optional redundancy summaries, plus adjustment conditioning.
        self.trim_global_feat_dim = feature_dims(self.feature_set)["global"]
        self.trim_action_feat_dim = (
            self.trim_base_feat_dim +
            self.trim_overlap_feat_dim +
            self.trim_redundancy_feat_dim +
            self.trim_global_feat_dim
        )
        self.forbid_halt_when_overlong = forbid_halt_when_overlong
        trim_scorer_indim = self.full_nodepair_dim + self.trim_action_feat_dim
        self.trim_scorer = nn.Sequential(
            FeatureNorm(trim_scorer_indim),
            get_mlp(n_trim_scorer_layers, trim_scorer_hidden_dim,
                    self.nonlin_type, self.dropout,
                    in_dim=trim_scorer_indim, out_dim=1)
        )


    def _decode_actions(self, state, greedy=False,
                          actions=None, precalc_data=None,
                          action_kinds=None, *, mask):
        """Take one typed route action for the given state."""
        log.debug("stepping with trim actions")
        if not isinstance(state, Observation):
            raise TypeError("policy execution requires an Observation")

        if precalc_data is None:
            node_descs, node_pad_mask, np_embeds, path_scores = \
                self._encode_graph(state)
        else:
            node_descs, node_pad_mask, np_embeds, path_scores = precalc_data

        batch_size = state.batch_size
        path_seqs = state.get_shortest_path_sequences()
        path_lens, current_route_lens, paths_are_invalid = path_mask(
            state, path_seqs, self.max_act_len, self.force_linking_unlinked)
        path_scores = set_invalid_scores_to_fmin(path_scores,
                                                      ~paths_are_invalid)
        trim_np_embeds = np_embeds

        starting = ~state.is_done() & (current_route_lens == 0)

        padding = (0, 1, 0, 1, 0, 0)
        drive_times = torch.nn.functional.pad(state.drive_times, padding)
        padding = (0, 0) + padding
        np_embeds = torch.nn.functional.pad(np_embeds, padding)
        are_on_routes = torch.zeros((batch_size, state.max_n_nodes + 1),
                                    device=state.device).bool()
        batch_idxs = state.batch_indices
        are_on_routes[batch_idxs[:, None], state.current_routes] = True
        are_on_routes[:, -1] = False
        getext_args = (state, current_route_lens, are_on_routes, path_seqs,
                       np_embeds, drive_times, path_scores, path_lens,
                       paths_are_invalid)
        prev_scores, prev_valid = self._get_extension_scores(
            *getext_args, before_or_after='before')
        next_scores, next_valid = self._get_extension_scores(
            *getext_args, before_or_after='after')

        ext_valid = prev_valid | next_valid
        ext_scores = prev_scores * prev_valid + next_scores * next_valid + \
            TORCH_FMIN * ~ext_valid
        start_scores = self._update_path_scores(state, path_scores, path_lens,
                                                state.drive_times,
                                                candidate_paths=path_seqs)

        trim_start_scores, trim_start_valid = self._get_trim_action_scores(
            state, trim_np_embeds, trim_start=True)
        trim_end_scores, trim_end_valid = self._get_trim_action_scores(
            state, trim_np_embeds, trim_start=False)

        trim_start_scores, trim_start_valid, trim_end_scores, trim_end_valid = trim_coverage_mask(
            state, trim_start_scores, trim_start_valid, trim_end_scores, trim_end_valid,
            self.force_linking_unlinked)

        no_route_action_yet = torch.zeros((batch_size,), dtype=torch.bool,
                                          device=state.device)
        route_is_done = state.is_done()
        chose_halt = route_is_done.clone()
        corh_logit = torch.zeros((batch_size,), device=state.device)
        corh_ent = torch.zeros((batch_size,), device=state.device)

        extending = ~starting & ~route_is_done
        xtnding_exp = extending[:, None, None]
        ppe_scores = start_scores * ~xtnding_exp + ext_scores * xtnding_exp
        ppe_valid = (~paths_are_invalid) * ~xtnding_exp + ext_valid * xtnding_exp
        ppe_valid = ppe_valid.bool()

        flat_extend_scores = ppe_scores.reshape(batch_size, -1)
        flat_extend_valid = ppe_valid.reshape(batch_size, -1)
        flat_trim_start_scores = trim_start_scores.reshape(batch_size, -1)
        flat_trim_start_valid = trim_start_valid.reshape(batch_size, -1)
        flat_trim_end_scores = trim_end_scores.reshape(batch_size, -1)
        flat_trim_end_valid = trim_end_valid.reshape(batch_size, -1)

        flat_extend_scores, flat_extend_valid = \
            mask_action_family(
                flat_extend_scores, flat_extend_valid, mask.extend)
        flat_trim_start_scores, flat_trim_start_valid = \
            mask_action_family(
                flat_trim_start_scores, flat_trim_start_valid,
                mask.trim_start)
        flat_trim_end_scores, flat_trim_end_valid = \
            mask_action_family(
                flat_trim_end_scores, flat_trim_end_valid, mask.trim_end)

        flat_route_scores = torch.cat((
            flat_extend_scores,
            flat_trim_start_scores,
            flat_trim_end_scores,
        ), dim=-1)
        flat_route_valid = torch.cat((
            flat_extend_valid,
            flat_trim_start_valid,
            flat_trim_end_valid,
        ), dim=-1)
        no_route_action_yet = ~flat_route_valid.any(-1)

        halt_scores = self.halt_scorer(state, node_descs, state.current_routes,
                                       state.current_route_time, node_pad_mask)
        halt_scores, old_route_is_done = edit_halt_scores(
            state, halt_scores, current_route_lens, no_route_action_yet,
            trim_start_valid, trim_end_valid, self.force_linking_unlinked,
            self.forbid_halt_when_overlong, mask.halt)

        action_log_probs = _joint_route_action_log_probs(
            flat_route_scores, flat_route_valid, halt_scores,
            self.serial_halting, self.temperature, old_route_is_done)

        if self.serial_halting:
            continue_scores = -halt_scores
            cont_or_halt = torch.cat((continue_scores, halt_scores), dim=-1)

            given_halt_actions = None
            if actions is not None:
                action_kinds = self._coerce_action_kinds(actions, action_kinds,
                                                         state.device)
                given_halt_actions = \
                    (action_kinds == RouteActionKind.HALT).to(torch.long)

            halt, corh_logit, corh_ent = select(cont_or_halt, not greedy,
                                                self.temperature,
                                                selection=given_halt_actions)
            assert (corh_logit > TORCH_FMIN).all(), "halt score is too low!"
            corh_logit = corh_logit.squeeze(1)
            chose_halt = halt.squeeze(1).bool()
            route_is_done = old_route_is_done | chose_halt
        else:
            route_is_done = old_route_is_done

        if not self.serial_halting:
            flat_route_scores = torch.cat((flat_route_scores, halt_scores),
                                          dim=-1)
            halt_idx = flat_route_scores.shape[-1] - 1

        given_route_actions = None
        if actions is not None:
            action_kinds = self._coerce_action_kinds(actions, action_kinds,
                                                     state.device)
            given_route_actions = self._encode_route_action_selection(
                state, actions, action_kinds
            )
            if self.serial_halting:
                given_route_actions[route_is_done] = 0
            else:
                given_route_actions[route_is_done] = halt_idx
                given_route_actions[action_kinds == RouteActionKind.HALT] = halt_idx

        flat_idxs, route_logit, route_ent = select(
            flat_route_scores, not greedy, self.temperature,
            selection=given_route_actions
        )
        flat_idxs = flat_idxs.squeeze(-1)
        route_logit = route_logit.squeeze(-1)

        if self.serial_halting:
            route_logit[chose_halt] = 0.0
            route_ent[chose_halt] = 0.0
        else:
            chose_halt = flat_idxs == halt_idx
            route_is_done = route_is_done | chose_halt

        assert ((route_logit > TORCH_FMIN) | route_is_done).all()

        action_kinds, folded_idxs = self._decode_route_action_selection(
            state, flat_idxs
        )
        action_kinds[route_is_done] = RouteActionKind.HALT
        folded_idxs[route_is_done] = -1

        if self.serial_halting:
            logits = corh_logit + route_logit
            entropy = corh_ent + route_ent
        else:
            logits = route_logit
            entropy = route_ent

        return action_kinds, folded_idxs, logits, entropy, action_log_probs

    def _coerce_action_kinds(self, actions, action_kinds, device):
        if action_kinds is not None:
            return action_kinds.to(device=device, dtype=torch.long)

        inferred = torch.full((actions.shape[0],), RouteActionKind.EXTEND,
                              dtype=torch.long, device=device)
        inferred[actions[:, 0] < 0] = RouteActionKind.HALT
        return inferred

    def _encode_route_action_selection(self, state, actions, action_kinds):
        # Flat layout: [extend (N*N), trim_start (route_capacity),
        #               trim_end (route_capacity)]. extend candidates are node
        # pairs; trim candidates are ROUTE POSITIONS (actions[:, 0] = position).
        n_nodes = state.max_n_nodes
        n_node_pairs = n_nodes * n_nodes
        route_capacity = state.current_routes.shape[1]
        safe_actions = actions.clamp(min=0)
        selection = safe_actions[:, 0] * n_nodes + safe_actions[:, 1]
        ts = action_kinds == RouteActionKind.TRIM_START
        te = action_kinds == RouteActionKind.TRIM_END
        selection[ts] = n_node_pairs + safe_actions[ts, 0]
        selection[te] = n_node_pairs + route_capacity + safe_actions[te, 0]
        return selection

    def _decode_route_action_selection(self, state, flat_idxs):
        n_nodes = state.max_n_nodes
        n_node_pairs = n_nodes * n_nodes
        route_capacity = state.current_routes.shape[1]
        action_kinds = torch.full_like(flat_idxs, RouteActionKind.EXTEND)

        trim_start = (flat_idxs >= n_node_pairs) & \
            (flat_idxs < n_node_pairs + route_capacity)
        trim_end = (flat_idxs >= n_node_pairs + route_capacity) & \
            (flat_idxs < n_node_pairs + 2 * route_capacity)
        action_kinds[trim_start] = RouteActionKind.TRIM_START
        action_kinds[trim_end] = RouteActionKind.TRIM_END

        # extend -> (from_node, to_node); clamp keeps the divmod valid for the
        # (overwritten) trim rows.
        pair_idxs = flat_idxs.clamp(max=n_node_pairs - 1)
        from_idxs = torch.div(pair_idxs, n_nodes, rounding_mode='floor')
        to_idxs = pair_idxs % n_nodes
        folded_idxs = torch.stack((from_idxs, to_idxs), dim=-1)

        # trim -> (position, -1)
        pos_start = flat_idxs - n_node_pairs
        pos_end = flat_idxs - n_node_pairs - route_capacity
        folded_idxs[trim_start, 0] = pos_start[trim_start]
        folded_idxs[trim_start, 1] = -1
        folded_idxs[trim_end, 0] = pos_end[trim_end]
        folded_idxs[trim_end, 1] = -1
        return action_kinds, folded_idxs

    def _get_trim_action_scores(self, state, nodepair_embeds,
                                trim_start=True):
        """Positional trim scores: one candidate per route position.

        Returns ``scores`` and ``valid`` of shape ``[batch, route_capacity]``.
        Each position's endpoint-pair embedding is GATHERED from
        ``nodepair_embeds`` (instead of scattering candidates into an [N, N]
        grid). Scattering collided whenever a node repeated on the route, which
        made positional trims ambiguous / no-ops; gathering by position is
        unambiguous and avoids the wasted N**2 action grid.
        """
        batch_size = state.batch_size
        route_capacity = state.current_routes.shape[1]
        global_features = global_state_features(state,
            include_redundancy=self.use_redundancy_features,
            preference_goals=self.preference_goals)
        global_features = global_features.to(
            device=state.device,
            dtype=nodepair_embeds.dtype,
        )
        if global_features.shape[-1] != self.trim_global_feat_dim:
            raise ValueError(
                "Trim scorer expects "
                f"{self.trim_global_feat_dim} global features, got "
                f"{global_features.shape[-1]}"
            )
        candidate_features, candidate_valid, from_nodes, to_nodes = \
            trim_candidate_features(
                state, global_features, nodepair_embeds.dtype, trim_start,
                use_redundancy=self.use_redundancy_features,
                allow_below_min=self.allow_trim_below_min)
        candidate_valid = candidate_valid & (from_nodes >= 0) & (to_nodes >= 0)
        scores = torch.full((batch_size, route_capacity), TORCH_FMIN,
                            device=state.device, dtype=nodepair_embeds.dtype)
        if candidate_valid.any():
            batch_idxs = torch.arange(
                batch_size, device=state.device)[:, None].expand(
                    -1, route_capacity)
            safe_from = from_nodes.clamp(min=0)
            safe_to = to_nodes.clamp(min=0)
            # [batch, route_capacity, embed_dim]
            pair_embeds = nodepair_embeds[batch_idxs, safe_from, safe_to]
            trim_inputs = torch.cat((pair_embeds, candidate_features), dim=-1)
            scored = self.trim_scorer(trim_inputs[candidate_valid]).squeeze(-1)
            scores[candidate_valid] = scored
        return scores, candidate_valid


    def _write_trim_features(self, features, valid, batch_idx, from_node,
                             to_node, removed_len, remaining_len, route_len,
                             max_len, removed_time, remaining_time,
                             route_time, direction_flag, removed_nodes,
                             removed_edge_nodes, kept_nodes, kept_edge_nodes,
                             other_node_mask, other_edge_mask,
                             global_features):
        base_features = torch.stack((
            torch.as_tensor(removed_len / route_len,
                            device=features.device, dtype=features.dtype),
            torch.as_tensor(remaining_len / max_len,
                            device=features.device, dtype=features.dtype),
            torch.as_tensor(route_len / max_len,
                            device=features.device, dtype=features.dtype),
            removed_time / route_time,
            remaining_time / route_time,
            torch.as_tensor(direction_flag, device=features.device,
                            dtype=features.dtype),
        ))
        overlap_features = self._get_trim_overlap_features(
            removed_nodes, removed_edge_nodes, kept_nodes, kept_edge_nodes,
            other_node_mask, other_edge_mask, features.dtype
        )
        features[batch_idx, from_node, to_node] = torch.cat((
            base_features, overlap_features, global_features
        ))
        valid[batch_idx, from_node, to_node] = True


    def _get_trim_overlap_features(self, removed_nodes, removed_edge_nodes,
                                   kept_nodes, kept_edge_nodes,
                                   other_node_mask, other_edge_mask, dtype):
        removed_node_overlap = self._get_node_overlap_fraction(
            removed_nodes, other_node_mask, dtype)
        removed_edge_overlap = self._get_edge_overlap_fraction(
            removed_edge_nodes, other_edge_mask, dtype)
        kept_node_overlap = self._get_node_overlap_fraction(
            kept_nodes, other_node_mask, dtype)
        kept_edge_overlap = self._get_edge_overlap_fraction(
            kept_edge_nodes, other_edge_mask, dtype)

        one = torch.ones((), device=other_node_mask.device, dtype=dtype)
        return torch.stack((
            removed_node_overlap,
            one - removed_node_overlap,
            removed_edge_overlap,
            one - removed_edge_overlap,
            kept_node_overlap,
            one - kept_node_overlap,
            kept_edge_overlap,
            one - kept_edge_overlap,
        ))

    def _get_node_overlap_fraction(self, nodes, node_mask, dtype):
        if nodes.numel() == 0:
            return torch.zeros((), device=node_mask.device, dtype=dtype)
        return node_mask[nodes].to(dtype=dtype).mean()

    def _get_edge_overlap_fraction(self, route_nodes, edge_mask, dtype):
        if route_nodes.numel() < 2:
            return torch.zeros((), device=edge_mask.device, dtype=dtype)
        return edge_mask[route_nodes[:-1], route_nodes[1:]].to(
            dtype=dtype).mean()


# primitive modules


class GraphNetBase(nn.Module):
    def __init__(self, n_layers, embed_dim, in_node_dim=None, in_edge_dim=None,
                 out_dim=None, nonlin_type=DEFAULT_NONLIN,
                 dropout=ATTN_DEFAULT_DROPOUT, recurrent=False, residual=False,
                 dense=False, n_proj_layers=1, use_norm=False, layer_kwargs={}):
        """nonlin type: nonlinearity, or the string name the non-linearity to
            use."""
        super().__init__()

        # do some input checking
        assert not (residual and dense)
        assert not (recurrent and dense)

        # set up members
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.in_node_dim = embed_dim if in_node_dim is None else in_node_dim
        self.in_edge_dim = in_edge_dim
        self.out_dim = embed_dim if out_dim is None else out_dim
        self.recurrent = recurrent
        self.residual = residual
        self.dense = dense
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        if isinstance(nonlin_type, str):
            self.nonlin_type = getattr(nn, nonlin_type)
        else:
            self.nonlin_type = nonlin_type
        self.nonlin = self.nonlin_type()
        self.embed_dim = embed_dim
        self.layer_kwargs = layer_kwargs

        self.node_in_proj = nn.Identity()
        self.edge_in_proj = nn.Identity()
        self.node_out_proj = nn.Identity()
        self.edge_out_proj = nn.Identity()

        if recurrent:
            if self.in_node_dim != self.embed_dim:
                # project nodes to embedding dimension
                self.node_in_proj = nn.Linear(self.in_node_dim, self.embed_dim)
            if self.gives_edge_features and self.in_edge_dim != self.embed_dim:
                # project edges to embedding dimension
                self.edge_in_proj = nn.Linear(self.in_edge_dim, self.embed_dim)

        final_nodedim = self._get_final_nodedim()
        if self.out_dim != final_nodedim:
            # projected embedded nodes to output dimension
            self.node_out_proj = get_mlp(n_proj_layers, self.embed_dim,
                                         in_dim=final_nodedim,
                                         out_dim=self.out_dim)
            if self.gives_edge_features:
                self.edge_out_proj = get_mlp(n_proj_layers, self.embed_dim,
                                             in_dim=self._get_final_edgedim(),
                                             out_dim=self.out_dim)

        # regardless of whether the network is recurrent, we still use separate
         # normalization layers at each step
        self.use_norm = use_norm
        if use_norm:
            norm_layers = [GraphNorm(self.embed_dim)
                        for _ in range(self.n_layers - 1)]
            self.node_norm_layers = nn.ModuleList(norm_layers)
            if self.gives_edge_features:
                norm_layers = [BatchNorm(self.embed_dim)
                            for _ in range(self.n_layers - 1)]
                self.edge_norm_layers = nn.ModuleList(norm_layers)

    def _get_layer_node_indims(self):
        return _layer_indims_helper(self.in_node_dim, self.embed_dim,
                                    self.n_layers, self.dense)

    def _get_layer_edge_indims(self):
        return _layer_indims_helper(self.in_edge_dim, self.embed_dim,
                                    self.n_layers, self.dense)

    def _get_final_nodedim(self):
        indims = self._get_layer_node_indims()
        if self.dense:
            return indims[-1] + self.embed_dim
        else:
            return self.embed_dim

    def _get_final_edgedim(self):
        indims = self._get_layer_edge_indims()
        if self.dense:
            return indims[-1] + self.embed_dim
        else:
            return self.embed_dim

    @property
    def gives_edge_features(self):
        raise NotImplementedError

    def forward(self, data):
        data = self.preprocess(data)
        if self.gives_edge_features:
            output = self._forward_helper(data)
        else:
            output = self._forward_helper(data), None
        return self.postprocess(*output)

    def preprocess(self, data):
        data.x = self.node_in_proj(data.x)
        if data.edge_attr is not None:
            data.edge_attr = self.edge_in_proj(data.edge_attr)
        return data

    def postprocess(self, node_embeds, edge_embeds):
        node_embeds = self.node_out_proj(node_embeds)
        if edge_embeds is not None:
            edge_embeds = self.edge_out_proj(edge_embeds)
        if self.gives_edge_features:
            return node_embeds, edge_embeds
        else:
            return node_embeds

    def _forward_helper(self, data):
        node_descs = data.x
        edge_descs = data.edge_attr

        for li, layer in enumerate(self.layers):
            # apply the layer and graph normalization
            out = layer(node_descs, data.edge_index, edge_descs)
            if self.gives_edge_features:
                layer_nodes, layer_edges = out
            else:
                layer_nodes, layer_edges = out, None

            if li < len(self.layers) - 1:
                # this is not the last layer, so apply dropout and nonlinearity
                if self.use_norm:
                    layer_nodes = self.node_norm_layers[li](layer_nodes,
                                                            data.batch)
                layer_nodes = self.nonlin(self.dropout(layer_nodes))
                if layer_edges is not None:
                    if self.use_norm:
                        layer_edges = self.edge_norm_layers[li](layer_edges)
                    layer_edges = self.nonlin(self.dropout(layer_edges))

            # get next layer's inputs based on this layer's outputs
            if self.residual and node_descs.shape[-1] == layer_nodes.shape[-1]:
                # add the input to the output
                node_descs = node_descs + layer_nodes
                if layer_edges is not None:
                    edge_descs = edge_descs + layer_edges
            elif self.dense:
                # concatenate the input to the output
                node_descs = torch.cat((node_descs, layer_nodes), dim=-1)
                if layer_edges is not None:
                    edge_descs = torch.cat((edge_descs, layer_edges), dim=-1)
            else:
                # just replace the input with the output
                node_descs = layer_nodes
                if layer_edges is not None:
                    edge_descs = layer_edges

        assert node_descs.shape[-1] == self._get_final_nodedim()

        # return the outputs
        if not self.gives_edge_features:
            # don't return the same edge features
            return node_descs
        else:
            return node_descs, edge_descs





class GraphAttnNet(GraphNetBase):
    def __init__(self, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        assert self.embed_dim % n_heads == 0, \
            'embed_dim is not divisible by n_heads!'
        head_dim = self.embed_dim // self.n_heads
        make_layer = lambda in_dim: \
            GATv2Conv(in_dim, head_dim, self.n_heads,
                      edge_dim=self.in_edge_dim, **self.layer_kwargs)

        if self.recurrent:
            layers = [make_layer(self.embed_dim)] * self.n_layers
        else:
            in_dims = self._get_layer_node_indims()
            layers = [make_layer(in_dim) for in_dim in in_dims]

        self.layers = nn.ModuleList(layers)

    @property
    def gives_edge_features(self):
        return False


# helper functions

def get_node_pair_descs(node_descs):
    n_nodes = node_descs.shape[-2]
    exp_dst_nodes = node_descs[:, None].expand(-1, n_nodes, -1, -1)
    exp_src_nodes = exp_dst_nodes.permute(0, 2, 1, 3)
    return torch.cat((exp_src_nodes, exp_dst_nodes), dim=-1)


def _layer_indims_helper(in_dim, embed_dim, n_layers, is_dense):
    indims = []
    if is_dense:
        curdim = in_dim
        for _ in range(n_layers):
            indims.append(curdim)
            curdim += embed_dim
    else:
        indims = [in_dim]
        indims += [embed_dim] * (n_layers - 1)
    return indims


def _joint_route_action_log_probs(scores, valid, halt_scores, serial_halting,
                                 temperature, force_halt):
    """Log probabilities of all route actions, with halt in the last column.

    Serial halting is a joint distribution: P(action) = P(continue) *
    P(action | continue). Rows without route options are deterministic halts.
    This adds no parameters and leaves the action sampler/checkpoints intact.
    """
    scores = scores.masked_fill(~valid, -torch.inf)
    no_options = ~valid.any(-1)
    forced = force_halt | no_options | (halt_scores.squeeze(-1) == TORCH_FMAX)
    # Avoid log-softmax over an all-masked row, even when it is later replaced.
    safe_scores = torch.where(no_options[:, None], torch.zeros_like(scores), scores)
    if serial_halting:
        route_log_probs = logsoftmax(safe_scores, temperature).to(torch.float64)
        decision = logsoftmax(torch.cat((-halt_scores, halt_scores), -1),
                              temperature).to(torch.float64)
        log_probs = torch.cat((route_log_probs + decision[:, :1],
                               decision[:, 1:]), -1)
    else:
        log_probs = logsoftmax(torch.cat((safe_scores, halt_scores), -1),
                               temperature).to(torch.float64)
    log_probs = log_probs.masked_fill(
        torch.cat((~valid, halt_scores <= TORCH_FMIN), -1), -torch.inf)
    deterministic = torch.full_like(log_probs, -torch.inf)
    deterministic[:, -1] = 0.
    return torch.where(forced[:, None], deterministic, log_probs)


def select(scores, scores_as_logits=False, softmax_temp=1,
           n_selections=1, selection=None):
    """scores: an n_options or batch_size x n_options tensor."""
    if scores.ndim == 1:
        # add a batch dimension
        scores = scores[None, :]
    logits = logsoftmax(scores, softmax_temp).to(dtype=torch.float64)

    # do this to avoid nans from 0 * -inf
    ent_logits = get_update_at_mask(logits, logits.isinf(), 0)
    entropy = -(ent_logits.exp() * ent_logits).sum(dim=-1)
    assert not entropy.isnan().any()

    if selection is None:
        if not scores_as_logits:
            _, selection = logits.topk(n_selections)
        else:
            probs = logits.exp()
            selection = probs.multinomial(n_selections)
            # if fewer than n_selections options survive rounding down by exp()
             # then use topk to select the top n_selections
            needs_topk = (probs > 0).sum(dim=1) < n_selections
            if needs_topk.any():
                _, topk_selection = logits[needs_topk].topk(n_selections)
                selection[needs_topk] = topk_selection

        selection.squeeze_(-1)

    if n_selections == 1:
        # add a "selection" dimension
        selection = selection[..., None]
        if scores.ndim == 1:
            selected_logits = logits[selection]
        else:
            # there is a batch dimension
            selected_logits = logits.gather(1, selection)

    else:
        # renormalize the log-probabilities after each selection
        idxs = torch.arange(logits.shape[1]).expand(logits.shape[0], -1)
        idxs = idxs.to(scores.device)
        selection_mask = idxs[..., None] == selection[:, None]
        selection_mask = selection_mask.any(dim=-1)
        selection_neutralizer = torch.zeros_like(scores)
        selection_neutralizer[selection_mask] = TORCH_FMIN
        unselected_scores = scores + selection_neutralizer
        if scores.ndim == 1:
            selected_scores = scores[selection]
        else:
            # there is a batch dimension
            selected_scores = scores.gather(1, selection)

        # flip the selected scores so the first selected comes last
        reordered_scores = \
            torch.cat((unselected_scores, selected_scores.flip(-1)), dim=-1)
        # sum them so the last element is all scores, 2nd last is all but
         # first picked, etc.
        logsumexps = torch.logcumsumexp(reordered_scores, -1)
        # flip again so the first selected comes first
        logdenoms = logsumexps[..., -n_selections:].flip(-1)
        selected_logits = selected_scores - logdenoms

    return selection, selected_logits, entropy


def logsoftmax(energies, softmax_temp=1):
    return nn.functional.log_softmax(energies / softmax_temp, dim=-1)


def get_mlp(n_layers, embed_dim, nonlin_type=DEFAULT_NONLIN,
            dropout=MLP_DEFAULT_DROPOUT, in_dim=None, out_dim=None, bias=True):
    """n_layers is the number of linear layers, so the number of 'hidden
    layers' is n_layers - 1."""
    layers = []
    for li in range(n_layers):
        if li == 0 and in_dim is not None:
            layer_in_dim = in_dim
        else:
            layer_in_dim = embed_dim
        if li == n_layers - 1 and out_dim is not None:
            layer_out_dim = out_dim
        else:
            layer_out_dim = embed_dim
        layers.append(nn.Linear(layer_in_dim, layer_out_dim, bias=bias))
        if li < n_layers - 1:
            layers.append(nn.Dropout(dropout))
            layers.append(nonlin_type())
    return nn.Sequential(*layers)
