"""Additional upstream encoders, scorers, graph nets and legacy generators.

Import these classes explicitly. This module depends on models; the active
models module does not import it. Scorers can be supplied as halt_scorer to
PathCombiningRouteGenerator without changing its parameter names.
"""
import logging as log
import math
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, MessagePassing, SGConv

from ..city.graph import DEMAND_KEY
from ..city.shortest_paths import reconstruct_all_paths
from ..common.tensors import get_update_at_mask, get_variable_slice_mask
from ..env.features import Observation, aggr_edges_over_sequences, PREFERENCE_KEYS
from ..env.masks import TORCH_FMIN, TORCH_FMAX
from ..network.routes import to_routes
from .const import DEFAULT_NONLIN, ATTN_DEFAULT_DROPOUT, MLP_DEFAULT_DROPOUT
from .models import (
    FeatureNorm, GraphAttnNet, GraphNetBase, PathCombiningRouteGenerator,
    RouteGeneratorBase, RouteScorer, TrimPathCombiningRouteGenerator,
    get_mlp, get_node_pair_descs, select,
)
from .trunc_normal import TruncatedNormal


PlanResults = namedtuple(
    "PlanResults",
    ["state", "stop_logits", "route_logits", "freq_logits",
    "entropy", "stops_tensor", "routes_tensor", "freqs_tensor", "stop_est_vals",
    "route_est_vals", "freq_est_vals"]
    )


RouteGenResults = namedtuple(
    "GenRouteResults", ["routes", "route_descs", "logits", "selections",
                        "est_vals"]
    )


RouteChoiceResults = namedtuple(
    "RouteChoiceResults", ["logits", "selections", "est_vals"]
    )


FreqChoiceResults = namedtuple(
    "FreqChoiceResults", ["freqs", "logits", "selections", "est_vals"]
    )


class SymmetricLogUnit(nn.Module):
    """For positive X, it's the positive component of log(X + 1).  For
    negative X, it's the negative of the positive component of log(-X + 1)."""
    def forward(self, xx):
        return symmetric_log_unit(xx)


def symmetric_log_unit(xx):
    """For positive X, it's the positive component of log(X + 1).  For
    negative X, it's the negative of the positive component of log(-X + 1)."""
    log_abs = torch.log(xx.abs() + 1)
    return log_abs * (xx > 0) - log_abs * (xx <= 0)


class GraphEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim


class KoolGraphEncoder(GraphEncoder):
    def __init__(self, in_dim, enc_dim, n_heads=8, n_layers=3, ff_dim=512):
        super().__init__(enc_dim)
        self.linear_embedding = nn.Linear(in_dim, enc_dim, bias=True)
        enc_layer = nn.TransformerEncoderLayer(enc_dim, n_heads, ff_dim)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

    def forward(self, node_features, adjacency_matrix=None, *args, **kwargs):
        """Adjacency matrix is treated as binary!"""
        node_features = self.linear_embedding(node_features)
        if adjacency_matrix is not None:
            if not isinstance(adjacency_matrix, torch.Tensor):
                adjacency_matrix = torch.tensor(adjacency_matrix,
                    dtype=torch.bool)
        node_features = node_features[:, None, :]
        encoded_feats = self.transformer(node_features, adjacency_matrix > 0)
        return encoded_feats.squeeze(dim=1)


class KoolNextNodeScorer(nn.Module):
    # This follows Kool et al 2019
    def __init__(self, embed_dim, clip_size=10):
        """
        embed_dim: the dimension of the embeddings.
        softmax_temp: the temperature of the softmax distribution over nodes.
            default is 1 (no change to energies).
        clip_size: maximum absolute value of scores that can be returned.
        """
        super().__init__()
        self.feature_dim = embed_dim
        self.clip_size = clip_size

        self.node_embedding_projector = nn.Linear(embed_dim, embed_dim*3,
                                                  bias=False)
        self.context_layer2_projector = nn.Linear(embed_dim, embed_dim)

        self.attn_embeddings = None

    def precompute(self, node_vecs):
        embeddings = self.node_embedding_projector(node_vecs)
        self.attn_embeddings = embeddings.chunk(3, dim=-1)

    def forward(self, context, mask=None):
        """
        context: the context vector specific to this step.
        mask: a mask that is True for invalid choices.
        """
        # context = context.reshape(1, -1)
        k_layer1, v_layer1, k_layer2 = self.attn_embeddings
        # perform the attention operations

        if k_layer1.ndim == 3: # and context.ndim == 2:
            # there's a batch dimension in the embeddings, so add an
            # appropriate dimension to the context tensor
            context = context[:, None, :]

        layer1_attn = torch.matmul(context, k_layer1.transpose(-2, -1))
        layer1_attn = layer1_attn / np.sqrt(k_layer1.size(-1))
        layer1_scores = torch.softmax(layer1_attn, dim=-1)
        context_layer2 = torch.matmul(layer1_scores, v_layer1)
        context_layer2 = self.context_layer2_projector(context_layer2)
        layer2_attn = torch.matmul(context_layer2, k_layer2.transpose(-2, -1))
        scores = layer2_attn / np.sqrt(k_layer2.size(-1))

        if self.clip_size:
            # limit the range of magnitudes of the energies with a tanh.
            scores = self.clip_size * torch.tanh(scores)

        if scores.dim() == 3:
            scores = scores.squeeze(-2)

        if mask is not None:
            # keep masked elements as -inf, so they'll have probability 0.
            scores[mask] = TORCH_FMIN

        return scores

    def reset(self):
        self.attn_embeddings = None


class RoutesEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode_nodepairs(self, nodepair_descs, routes_tensor, **encode_args):
        """
        nodepair_descs: batch_size x n_nodepairs x embed_dim tensor
        """
        route_feats, pad_mask = \
            aggr_edges_over_sequences(routes_tensor, nodepair_descs, 'concat')

        # flatten multiple batch dimensions
        n_batch_dims = routes_tensor.ndim - 1
        if n_batch_dims > 1:
            old_batch_dim = pad_mask.shape[:-1]
            route_feats = route_feats.flatten(0, n_batch_dims - 1)
            pad_mask = pad_mask.flatten(0, n_batch_dims - 1)

        # remove sequences that are all padding
        is_valid_seq = ~pad_mask.all(dim=-1)
        route_feats = route_feats[is_valid_seq]
        pad_mask = pad_mask[is_valid_seq]
        enc = self.encode(route_feats, pad_mask, embed_pos=False,
                          **encode_args)

        # add back locations for the all-padding sequences
        feat_dim = enc.shape[-1]
        full_enc_shape = (is_valid_seq.shape[0], feat_dim)
        full_enc = torch.zeros(full_enc_shape, dtype=enc.dtype,
                               device=enc.device)
        full_enc[is_valid_seq] = enc
        # add back multiple batch dimensions
        if n_batch_dims > 1:
            full_enc = full_enc.reshape(old_batch_dim + (feat_dim,))
        return full_enc

    def forward(self, node_descs, routes_tensor, padding_mask, **encode_args):
        # put everything on the right device
        dev = node_descs.device
        routes_tensor = routes_tensor.to(device=dev)
        padding_mask = padding_mask.to(device=dev)

        # build route tensor and mask
        old_shape = None
        nodes_have_batch = node_descs.ndim > 2
        if nodes_have_batch:
            batch_size = node_descs.shape[0]
            nodes_per_batch = node_descs.shape[-2]
            node_descs = node_descs.reshape(-1, node_descs.shape[-1])

            if routes_tensor.ndim == 2:
                # give the routes tensor a batch dim
                sizes = (batch_size,) + (-1,) * routes_tensor.ndim
                routes_tensor = routes_tensor[None]
                routes_tensor = routes_tensor.expand(sizes)

            if padding_mask.ndim == 2:
                # give the pad mask a batch dim
                sizes = (batch_size,) + (-1,) * padding_mask.ndim
                padding_mask = padding_mask[None]
                padding_mask = padding_mask.expand(sizes)

        if routes_tensor.ndim > 2:
            # we're dealing with multiple routes per batch element
            old_shape = routes_tensor.shape
            batch_size = old_shape[0]
            if nodes_have_batch:
                # add offsets to each batch element to the right set of nodes
                 # in the flattened node_descs
                offsets = torch.arange(0, batch_size * nodes_per_batch,
                                       nodes_per_batch,
                                       device=node_descs.device)
                reshape_dims = (-1,) + (1,) * (len(old_shape) - 1)
                routes_tensor = routes_tensor + offsets.reshape(reshape_dims)

            # flatten multiple batch dimensions
            routes_tensor = routes_tensor.reshape(-1, routes_tensor.shape[-1])
            padding_mask = padding_mask.reshape(-1, routes_tensor.shape[-1])

        # cut out invalid rows (padding mask is true everywhere), then add them
         # back
        enc_shape = (routes_tensor.shape[0], node_descs.shape[-1])
        valid_seqs = ~padding_mask.all(dim=1)
        routes_tensor = routes_tensor[valid_seqs]
        padding_mask = padding_mask[valid_seqs]

        # get node features for routes
        route_node_feats = node_descs[routes_tensor]
        encode_out = self.encode(route_node_feats, padding_mask, **encode_args)
        if isinstance(encode_out, tuple):
            valid_enc = encode_out[0]
        else:
            valid_enc = encode_out
        enc = torch.zeros(enc_shape, device=dev)
        enc[valid_seqs] = valid_enc
        # remove the sequence dimension in the seq-len-1 encoding
        enc = enc.squeeze(1)
        if old_shape is not None:
            enc = enc.reshape(old_shape[:-1] + (enc.shape[-1],))
        if isinstance(encode_out, tuple):
            return (enc,) + encode_out[1:]
        else:
            return enc

    def encode(self, route_seqs, padding_mask):
        raise NotImplementedError


class SumRouteEncoder(RoutesEncoder):
    def encode_nodepairs(self, nodepair_descs, routes_tensor, **encode_args):
        """
        nodepair_descs: batch_size x n_nodepairs x embed_dim tensor
        """
        route_feats = \
            aggr_edges_over_sequences(routes_tensor, nodepair_descs, 'sum')

        return route_feats


class MeanRouteEncoder(RoutesEncoder):
    def encode(self, route_seqs, padding_mask, *args, **kwargs):
        enc = mean_pool_sequence(route_seqs, padding_mask)
        return enc


class MaxRouteEncoder(RoutesEncoder):
    def encode(self, route_seqs, padding_mask, *args, **kwargs):
        unsq_pad = padding_mask.unsqueeze(-1)
        enc = route_seqs * ~unsq_pad + TORCH_FMIN * unsq_pad
        enc, _ = route_seqs.max(dim=1)
        return enc


class LatentAttnRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_heads=8, n_layers=2,
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        self.encoder = LatentAttentionEncoder(embed_dim, 1, n_heads, n_layers,
                                              dropout)
        self.end_token = nn.Parameter(torch.randn(embed_dim))

    def encode(self, route_seqs, padding_mask, embed_pos=True):
        # TODO somehow also encode the inter-stop times
        enc = self.encoder(route_seqs, padding_mask=padding_mask,
                           embed_pos=embed_pos).squeeze(1)
        return enc


class TransformerRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_heads, n_layers,
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(embed_dim, n_heads, 4*embed_dim,
                                               dropout=dropout,
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.end_token = nn.Parameter(torch.randn(embed_dim))
        # self.final_encoder = LatentAttentionEncoder(embed_dim, 1, n_heads, 1,
        #                                             dropout)

    def encode(self, route_seqs, padding_mask, embed_pos=True,
               return_nodes=False):
        # append the end token to each route sequence
        seq_lens = (~padding_mask).sum(dim=1)
        max_seq_len = seq_lens.max()
        end_token = self.end_token.expand(route_seqs.shape[0], -1)
        if max_seq_len + 1 > route_seqs.shape[1]:
            # pad the end of route seqs and padding mask
            placeholder = torch.zeros_like(end_token)[:, None]
            route_seqs = torch.cat((route_seqs, placeholder), dim=1)
            padding_mask = torch.cat((padding_mask,
                                      torch.ones((route_seqs.shape[0], 1),
                                                 dtype=bool)),
                                     dim=1)

        # insert end tokens
        batch_idxs = torch.arange(route_seqs.shape[0])
        route_seqs[batch_idxs, seq_lens] = end_token
        # set locations corresponding to new end token to False
        padding_mask[batch_idxs, seq_lens] = False

        if embed_pos:
            sinseq = get_sinusoid_pos_embeddings(route_seqs.shape[1],
                                                route_seqs.shape[2])
            sinseq = sinseq[None]
            route_seqs = route_seqs + sinseq.to(device=route_seqs.device)

        tfed = self.transformer(route_seqs, src_key_padding_mask=padding_mask)
        # take the embedding at the placeholder token as the descriptor
        route_descs = tfed[batch_idxs, seq_lens]
        if return_nodes:
            return route_descs, tfed
        else:
            return route_descs


class ConvRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_layers):
        super().__init__()
        layers = sum([[nn.Conv1d(embed_dim, embed_dim, 3),
                       DEFAULT_NONLIN(),
                       nn.AvgPool1d(2)] for _ in range(n_layers)], start=[])
        # applied to a boolean, this should act like "or" over each pooled
         # block
        self.mask_pooler = nn.MaxPool1d(2**n_layers)
        self.encoder = nn.Sequential(*layers)

    def encode(self, route_seqs, padding_mask):
        # swap sequence and channel dims, since that's what conv1d expects
        route_seqs = route_seqs.permute(0, 2, 1)
        conved = self.encoder(route_seqs)
        # swap sequence and channel dims back
        conved = conved.permute(0, 2, 1)

        # compute new padding mask
        valid_mask = (~padding_mask[:, None]).to(dtype=float)
        pooled_valid_mask = self.mask_pooler(valid_mask)
        pooled_valid_mask = pooled_valid_mask[:, 0].to(dtype=bool)
        # due to padding, the shapes might not match perfectly
        pooled_valid_mask = pooled_valid_mask[:, :conved.shape[1]]
        pooled_pad_mask = ~pooled_valid_mask

        # do mean-pooling over pooled sequences
        pooled = mean_pool_sequence(conved, pooled_pad_mask)
        return pooled


class ContinuousGaussianActor(nn.Module):
    def __init__(self, n_layers, embed_dim, in_dim=None,
                 min_action=None, max_action=None, min_logstd=None,
                 max_logstd=None, bias=True):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.mean_and_std_net = get_mlp(n_layers, embed_dim, in_dim=in_dim,
                                        out_dim=2, bias=bias)

    def forward(self, inpt, pick_max=False):
        means_and_stds = self.mean_and_std_net(inpt)
        means = means_and_stds[..., 0]
        if self.min_action is not None or self.max_action is not None:
            means = means.clamp(self.min_action, self.max_action)

        # stds must be positive, so exponentiate them
        logstds = means_and_stds[..., 1]
        if self.min_logstd is not None or self.max_logstd is not None:
            logstds = logstds.clamp(self.min_logstd, self.max_logstd)
        stds = torch.exp(logstds)

        if self.min_action is not None or self.max_action is not None:
            dstrb = TruncatedNormal(means, stds, self.min_action,
                                    self.max_action)
        else:
            dstrb = torch.distributions.Normal(means, stds)

        if pick_max:
            # take the action at the mean of the distribution
            actions = means
        else:
            # sample from the distribution and limit to valid range
            actions = dstrb.sample()

        log_probs = dstrb.log_prob(actions)
        return actions, log_probs


class ContinuousGaussFixedStdActor(nn.Module):
    def __init__(self, n_layers, embed_dim, in_dim=None,
                 min_action=None, max_action=None, fixed_std=None, bias=True):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        if fixed_std is None:
            if min_action is not None and max_action is not None:
                self.fixed_std = (max_action - min_action) / 5
            else:
                raise ValueError(
                    "Must provide fixed_std or min_action and max_action")
        else:
            self.fixed_std = fixed_std

        self.mean_net = get_mlp(n_layers, embed_dim, in_dim=in_dim, out_dim=1,
                                bias=bias)

    def forward(self, inpt, greedy=False):
        means = self.mean_net(inpt).squeeze(-1)
        means = means.clamp(self.min_action, self.max_action)
        stds = torch.full_like(means, self.fixed_std)

        dstrb = TruncatedNormal(means, stds, self.min_action, self.max_action)
        if greedy:
            # take the action at the mean of the distribution
            actions = means
        else:
            # sample from the distribution and limit to valid range
            actions = dstrb.sample()

        log_probs = dstrb.log_prob(actions)
        return actions, log_probs, stds


class FrequencySelector(nn.Module):
    def __init__(self, embed_dim, min_frequency_Hz=1/7200,
                 max_frequency_Hz=1/60):
        """default min_frequency_Hz is one bus every two hours.
           default max_frequency_Hz is one bus every minute."""
        super().__init__()
        self.mean_and_std_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            DEFAULT_NONLIN(),
            nn.Linear(embed_dim, embed_dim),
            DEFAULT_NONLIN(),
            nn.Linear(embed_dim, 2)
        )

        self.freq_embedder = nn.Sequential(
            nn.Linear(1, embed_dim),
            DEFAULT_NONLIN()
        )
        assert min_frequency_Hz < max_frequency_Hz
        # approximately 1.44 Hz
        self.min_logstd = -1
        self.min_action = self._freq_Hz_to_action(min_frequency_Hz)
        self.max_action = self._freq_Hz_to_action(max_frequency_Hz)
        # wide enough to give a uniform distribution
        self.max_logstd = np.log((self.max_action - self.min_action) * 10000)


    def forward(self, route_descriptor, greedy=False, rollout_freqs=None,
                min_frequency_Hz=None, max_frequency_Hz=None):
        assert not greedy or rollout_freqs is None, \
            "greedy frequency selection is incompatible with rollout!"
        # compute the parameters of the gaussians from which to sample freqs
        normal_param_tensor = self.mean_and_std_net(route_descriptor)
        # stop the mean from being lower than the minimum allowed action
        if min_frequency_Hz:
            min_action = self._freq_Hz_to_action(min_frequency_Hz)
        else:
            min_action = self.min_action
        if max_frequency_Hz:
            max_action = self._freq_Hz_to_action(max_frequency_Hz)
        else:
            max_action = self.max_action
        assert min_action < max_action

        means = normal_param_tensor[..., 0].clamp(min_action, max_action)
        # clamp the std devs in a sensible range for numerical stability
        logstds = normal_param_tensor[..., 1].clamp(self.min_logstd,
                                                    self.max_logstd)
        # stds must be positive, so exponentiate them
        stds = torch.exp(logstds)
        dstrb = torch.distributions.Normal(means, stds)
        if greedy:
            # greedily take the action at the center of the distribution
            actions = means
        elif rollout_freqs is None:
            # sample the action from the distribution
            actions = dstrb.sample()
        else:
            # take the dictated action
            actions = self._freq_Hz_to_action(rollout_freqs)

        # enforce the minimum allowable frequency for a route that's included
        actions.clamp_(self.min_action, self.max_action)
        log_probs = dstrb.log_prob(actions)

        freqs_Hz = self._action_to_freq_Hz(actions)
        descriptors = self.freq_embedder(actions[:, None])
        return freqs_Hz, log_probs, descriptors

    def _freq_Hz_to_action(self, freq_Hz):
        # the "action" is the log of the per-hour frequency.
        freq_hourly = freq_Hz * 3600
        if isinstance(freq_hourly, torch.Tensor):
            return torch.log(freq_hourly)
        else:
            return np.log(freq_hourly)

    def _action_to_freq_Hz(self, action):
        if isinstance(action, torch.Tensor):
            freq_hourly = torch.exp(action)
        else:
            freq_hourly = np.exp(action)
        return freq_hourly / 3600


class NodepairDotScorer(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_transform = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_transform = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, flat_node_vecs, batch_n_nodes=None):
        queries = self.query_transform(flat_node_vecs)
        queries /= math.sqrt(self.embed_dim)
        keys = self.key_transform(flat_node_vecs)
        dev = flat_node_vecs.device

        if batch_n_nodes is not None:
            # fold the queries and keys into a batch dimension
            max_n_nodes = max(batch_n_nodes)
            batch_size = len(batch_n_nodes)
            folded_queries = torch.zeros((batch_size, max_n_nodes,
                                          self.embed_dim), device=dev)
            folded_keys = folded_queries.clone()
            for bi, num_nodes in enumerate(batch_n_nodes):
                folded_queries[bi, :num_nodes, :queries.shape[-1]] = \
                    queries[:num_nodes]
                folded_keys[bi, :num_nodes, :keys.shape[-1]] = \
                    keys[:num_nodes]

        else:
            folded_queries = queries[None].contiguous()
            folded_keys = keys[None].contiguous()

        # because the folded tensors are zero everywhere there is no real
         # input, we don't need to mask anything.
        dot_prod = torch.bmm(folded_queries, folded_keys.transpose(-2, -1))
        return dot_prod


class GlobalMeanNodesRouteScorer(RouteScorer):
    @property
    def in_dim(self):
        return self.n_extra_feats + self.embed_dim

    def encode(self, node_descs, route_seqs, node_pad_mask=None,
               route_pad_mask=None, embed_pos=False):
        return torch.zeros((node_descs.shape[0], 0), device=node_descs.device)


class ValueNetScorer(GlobalMeanNodesRouteScorer):
    def forward(self, state, node_descs, node_pad_mask):
        time_placeholder = torch.zeros((state.batch_size), device=state.device)
        route_placeholder = torch.full((state.batch_size, state.max_n_nodes),
                                       -1, dtype=int, device=state.device)
        return super().forward(state, node_descs, route_placeholder,
                               time_placeholder, node_pad_mask)


class RouteUniformScorer(RouteScorer):
    def __init__(self, value=0.0, *args, **kwargs):
        super(RouteScorer, self).__init__()
        self.value = value

    def forward(self, state, node_descs, *args, **kwargs):
        batch_size = node_descs.shape[0]
        return torch.full((batch_size, 1), self.value,
                          device=node_descs.device)


class RouteAlphaScorer(RouteScorer):
    def __init__(self, *args, **kwargs):
        super(RouteScorer, self).__init__()

    def forward(self, state, node_descs, *args, **kwargs):
        if isinstance(state, Observation):
            column = state.feature_set.global_features.index("pref.RTT")
            route_time_weight = state.global_features[:, column]
        else:
            route_time_weight = state.cost_weights['route_time_weight']
        # this score gives halting probability = route_time_weight
        halt_score = ((1 - route_time_weight) / route_time_weight).log() / -2
        # clamp these to our torch fmin and fmax
        halt_score.clamp_(TORCH_FMIN, TORCH_FMAX)
        return halt_score[:, None]


class RouteMeanScorer(RouteScorer):
    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask):
        return mean_pool_sequence(route_seqs, route_pad_mask)


class RouteTransformerScorer(RouteScorer):
    def __init__(self, n_heads, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = TransformerRouteEncoder(self.embed_dim, n_heads,
                                               n_layers, self.dropout)

    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask,
               embed_pos=True):
        route_descs = self.encoder.encode(route_seqs, route_pad_mask,
                                          embed_pos)
        return route_descs


class RouteLatentScorer(RouteScorer):
    def __init__(self, n_heads, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = LatentAttentionEncoder(self.embed_dim, n_heads=n_heads,
                                              n_layers=n_layers,
                                              dropout=self.dropout)

    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask,
               embed_pos=True):
        query = mean_pool_sequence(node_descs, node_pad_mask)[:, None]
        enc = self.encoder(route_seqs, query, route_pad_mask, embed_pos)
        return enc.squeeze(1)


class DummyMLP(nn.Module):
    def __init__(self, out_dim, out_val=0):
        super().__init__()
        self.out_dim = out_dim
        self.out_val = out_val

    def forward(self, inpt):
        out_shape = tuple(inpt.shape[:-1]) + (self.out_dim,)
        output = torch.full(out_shape, self.out_val, device=inpt.device)
        return output


class RandomPathCombiningRouteGenerator(PathCombiningRouteGenerator):
    """Behaves like PathCombiningRouteGenerator, but all options are equally
        likely."""
    def __init__(self, halt_prob_is_route_time_weight=False,
                 *args, **kwargs):
        super().__init__(n_nodepair_layers=0, *args, **kwargs)
        # 0 so adding them gives the same result no matter how many are summed
        self.nodepair_scorer = DummyMLP(1, 0.0)
        self.path_scorer = DummyMLP(1, 1.0)
        if halt_prob_is_route_time_weight:
            self.halt_scorer = RouteAlphaScorer()
        else:
            # halting and continuing are equally likely
            self.halt_scorer = RouteUniformScorer()


class RandomTrimExtendRouteGenerator(TrimPathCombiningRouteGenerator):
    """Trim/extend edit generator with uniform scores for all valid actions."""

    def __init__(self, halt_prob_is_route_time_weight=False,
                 *args, **kwargs):
        super().__init__(n_nodepair_layers=0, *args, **kwargs)
        # Keep the same valid-action masks as the trim/extend model, but remove
        # learned scoring. BCO's outer greedy acceptance still decides whether
        # the sampled edit survives.
        self.nodepair_scorer = DummyMLP(1, 0.0)
        self.path_scorer = DummyMLP(1, 1.0)
        self.trim_scorer = DummyMLP(1, 1.0)
        if halt_prob_is_route_time_weight:
            self.halt_scorer = RouteAlphaScorer()
        else:
            self.halt_scorer = RouteUniformScorer()


class LegacyRouteGeneratorBase(RouteGeneratorBase):
    """Upstream loop that normalizes and mutates State, for legacy generators.

    step adds one route; forward builds the whole network and returns PlanResults.
    Active construction/edit policies use act/evaluate and Env transitions.
    """
    def plan(self, *args, **kwargs):
        return self.forward_oldenv(*args, **kwargs)

    def setup_planning(self, state):
        # apply normalization to input features
        from ..env.base import normalize_nodes
        normalize_nodes(state, self.node_norm, self.feature_set)
        return state

    def step(self, state, greedy):
        log.debug("stepping")
        # apply normalization to input features
        norm_stops_x = state.norm_node_features
        # assemble route data objects
        input_data_list = state.graph_data.to_data_list()
        for dd in input_data_list:
            # create a fully-connected graph with node embeddings
            dd.edge_index = dd[DEMAND_KEY].edge_index
            dd.x = norm_stops_x[:dd.num_nodes]
            norm_stops_x = norm_stops_x[dd.num_nodes:]

        # get edge features and assign them to the batch
        if isinstance(self.backbone_net, GraphAttnNet) or self.backbone_net.gives_edge_features:
            edge_features_square = self._get_edge_features(state)
            for ef, dd in zip(edge_features_square, input_data_list):
                dd.edge_attr = ef[dd.edge_index[0], dd.edge_index[1]]
        else:
            edge_features_square = None

        input_batch = Batch.from_data_list(input_data_list)

        # run GNN forward
        if self.backbone_net.gives_edge_features:
            route_node_embeds, _ = self.backbone_net(input_batch)
        else:
            route_node_embeds = self.backbone_net(input_batch)

        # do routing and choosing (different subclasses do it differently)
        log.debug("get routes")
        batch_new_routes, logits, entropy = self._get_routes(
            1, state, route_node_embeds, edge_features_square, greedy)

        # update the state
        state.add_new_routes(
            batch_new_routes, self.only_routes_with_demand_are_valid)
        return state, logits, entropy

    def forward(self, state, greedy=False):
        # do a full rollout
        state = self.setup_planning(state)

        log.debug("starting route-generation loop")
        all_logits = []
        all_entropy = 0
        for _ in range(state.n_routes_to_plan):
            state, logits, entropy = self.step(state, greedy)
            all_logits.append(logits)
            all_entropy += entropy

        routes_tensor = \
            to_routes(state.routes).to(state.device)
        # TODO make this return a named tuple with names that actually make
         # sense
        logits = torch.cat(all_logits, dim=1)
        result = PlanResults(
            state=state, stop_logits=None,
            route_logits=logits, freq_logits=None, entropy=entropy,
            stops_tensor=None, routes_tensor=routes_tensor, freqs_tensor=None,
            stop_est_vals=None, route_est_vals=None, freq_est_vals=None
        )
        return result

    def get_biased_scores(self, scores, bias, are_connected):
        # no bias for demand that's already covered
        # bias = get_update_at_mask(bias, are_connected)
        scores += bias
        # # set scores for invalid connections to 0
        scores = scores * ~are_connected

        return scores


class UnbiasedPathCombiner(LegacyRouteGeneratorBase):
    def __init__(self, *args, n_heads=1, n_encoder_layers=1,
                 n_selection_attn_layers=1, **kwargs):
        super().__init__(*args, **kwargs)
        n_extra_path_feats = 2
        self.path_extras_norm = FeatureNorm(n_extra_path_feats)
        # self.path_encoder = MeanRouteEncoder()
        # self.path_encoder = MaxRouteEncoder()
        # self.path_encoder = SumRouteEncoder()
        # self.path_encoder = LatentAttnRouteEncoder(self.embed_dim,
        #                                            n_heads, n_encoder_layers,
        #                                            self.dropout)
        self.path_encoder = TransformerRouteEncoder(self.embed_dim,
                                                    n_heads, n_encoder_layers,
                                                    self.dropout)
        self.nodepair_embedder = get_mlp(
            self.n_nodepair_layers, self.embed_dim, self.nonlin_type,
            self.dropout, in_dim=self.full_nodepair_dim)
        self.path_mlp = get_mlp(
            3, self.embed_dim * 2, self.nonlin_type, self.dropout,
            in_dim=self.embed_dim + n_extra_path_feats,
            out_dim=self.embed_dim)

        init_ctxt = nn.Parameter(torch.randn(1, self.embed_dim))
        self.register_parameter(name="init_ctxt", param=init_ctxt)

        self.n_selection_layers = n_selection_attn_layers
        self.kv_embedder = nn.Linear(
            self.embed_dim, self.embed_dim * (2 * n_selection_attn_layers + 1)
        )

        n_extra_global_feats = 5
        self.global_extras_norm = FeatureNorm(n_extra_global_feats)
        init_dim = self.embed_dim * 2 + n_extra_global_feats
        query_embedders = [nn.Linear(init_dim, self.embed_dim)]
        query_embedders += [nn.Linear(self.embed_dim, self.embed_dim)
                            for _ in range(n_selection_attn_layers - 1)]
        self.query_embedders = nn.ModuleList(query_embedders)

        self.halt_scorer = get_mlp(2, self.embed_dim * 2, self.nonlin_type,
                                   self.dropout, in_dim=self.embed_dim,
                                   out_dim=1)

    def _get_routes(self, n_routes, state, node_descs, extra_nodepair_feats,
                    greedy):
        """
        n_routes: int
        node_descs: batch_size x n_nodes x node_feat_dim
        state: batch_size x n_nodes x n_nodes x state_dim
        greedy: bool
        returns: batch_size x n_routes x route_len, all_logits, None
        """
        log.debug("get routes")
        # doesn't support chunk sizes greater than 1
        assert n_routes == 1

        dev = state.device

        # fold node descriptors
        node_descs, node_pad_mask = self.fold_node_descs(node_descs, state)
        nodepair_descs = get_node_pair_descs(node_descs)

        # concatenate the extra nodepair features if they're there
        if extra_nodepair_feats is not None:
            nodepair_descs = torch.cat((extra_nodepair_feats, nodepair_descs),
                                       dim=-1)
        nodepair_descs = self.nodepair_embedder(nodepair_descs)

        if '_seqs' not in state.graph_data.__dict__:
            # compute the shortest paths and store them in the graph_data object
             # so we don't need to recompute them for each route.
            seqs, _ = reconstruct_all_paths(state.graph_data.nexts)
            state.graph_data._seqs = seqs
        else:
            seqs = state.graph_data._seqs

        # encode the paths
        # with torch.no_grad():
        path_encs = self.encode_path(nodepair_descs, seqs, state.drive_times)
        path_attn_embeds = self.kv_embedder(path_encs)

        # do all the key and value embeddings at once
        base_ext_mask = ~state.valid_terms_mat.clone()
        ext_mask = base_ext_mask.flatten(1, 2)

        # set the initial context
        global_feats = torch.cat(
            ((state.total_route_time / state.n_routes_to_plan)[..., None],
             state.get_n_routes_features(),
             state.get_cost_weights_tensor(tuple(PREFERENCE_KEYS[goal] for goal in self.preference_goals))), dim=-1)
        global_feats = self.global_extras_norm(global_feats)
        # get average node descriptor
        avg_node = mean_pool_sequence(node_descs, node_pad_mask)
        # concatenate the above
        global_feats = torch.cat((global_feats, avg_node), dim=-1)
        init_ctxt = self.init_ctxt.expand(state.batch_size, -1).to(device=dev)
        context = torch.cat((global_feats, init_ctxt), dim=-1)

        # set up loop-tracking variables and other needed values
        route = torch.full((state.batch_size, state.max_n_nodes), -1,
                            device=dev)
        is_done = torch.zeros(state.batch_size, device=dev).bool()
        logits = None
        candidates = seqs.flatten(1, 2)
        cdt_times = state.drive_times.flatten(1, 2)
        first_iter = True
        batch_idxs = state.batch_indices
        relevant_attn_embeds = path_attn_embeds.flatten(1, 2)
        node_on_route = torch.zeros((state.batch_size, state.max_n_nodes + 1),
                                    device=dev).bool()
        route_times = torch.zeros(state.batch_size, device=dev)
        route_len = (route > -1).sum(-1)

        # assemble a route!
        while not is_done.all():
            # compute scores vs current context and pick an extension path
            out_context, ext_probs = self._get_extension_scores(
                state, context, relevant_attn_embeds, ext_mask)
            # zero out scores for paths that are too long
            cdt_lens = (candidates != -1).sum(dim=-1) + route_len[:, None]
            cdt_too_long = cdt_lens > state.max_route_len[:, None]
            ext_probs = get_update_at_mask(ext_probs, cdt_too_long)
            no_valid_ext = (ext_probs.sum(dim=-1) == 0)
            is_done |= no_valid_ext

            if not first_iter:
                # decide whether to halt using context
                halt_score = self.halt_scorer(out_context)
                halt_score[is_done] = TORCH_FMAX
                halt_score[route_len < state.min_route_len] = TORCH_FMIN
                cont_score = -halt_score
                cont_or_halt = torch.cat((cont_score, halt_score), dim=-1)
                halt, corh_logit = select(cont_or_halt, not greedy)
                # zero out logits where we've already halted
                corh_logit = corh_logit.squeeze(1) * ~is_done
                # update doneness
                is_done = is_done | halt.squeeze(1).bool()
                logits += corh_logit

            if greedy:
                selection = ext_probs.argmax(dim=-1)
            else:
                # avoid crash for all-zero entries
                ext_probs[no_valid_ext] = 1.0
                selection = ext_probs.multinomial(1)
            step_logit = ext_probs.gather(1, selection).log().squeeze(1)
            step_logit = step_logit + ~is_done
            if first_iter:
                logits = step_logit
            else:
                logits += step_logit + corh_logit

            # update the routes
            route_len = (route > -1).sum(-1)
            candidate_lens = (candidates > -1).sum(-1)
            chosen_ext_lens = candidate_lens.gather(1, selection).squeeze(1)
            # enforce no extension if is_done
            chosen_ext_lens[is_done] = 0
            # compute mask for new stops on the routes
            insert_mask = get_variable_slice_mask(
                route, dim=1, froms=route_len, tos=route_len + chosen_ext_lens)
            # select the chosen extensions
            exp_sel = selection[..., None].expand(-1, -1, candidates.shape[-1])
            chosen_seqs = candidates.gather(1, exp_sel).squeeze(1)
            # insert the chosen extensions
            select_mask = get_variable_slice_mask(
                chosen_seqs, dim=1, tos=chosen_ext_lens)
            # route[insert_mask] = chosen_seqs[select_mask]
            chosen_seqs_inplace = torch.zeros_like(route)
            chosen_seqs_inplace[insert_mask] = chosen_seqs[select_mask]
            route = route * ~insert_mask + chosen_seqs_inplace * insert_mask
            # update which nodes are on the route
            node_on_route[batch_idxs[:, None], chosen_seqs] = True
            # make padding column False
            node_on_route[:, -1] = False
            route_times += cdt_times.gather(1, selection).squeeze(1)

            # update the extension mask based on the current route
            # make only segments continuing from this one valid
            route_len += chosen_ext_lens
            end_nodes = route.gather(1, route_len[..., None] - 1).squeeze()
            relevant_attn_embeds = path_attn_embeds[batch_idxs, end_nodes]
            ext_mask = base_ext_mask[batch_idxs, end_nodes]
            # trim the first node of each candidate, since it's the last node
             # on the current route
            candidates = seqs[batch_idxs, end_nodes][..., 1:]
            cdt_times = state.drive_times[batch_idxs, end_nodes]
            # invalidate segments that revisit stops on the route
            revisits = node_on_route[batch_idxs[:, None, None], candidates]
            revisits = revisits.any(-1)
            ext_mask = ext_mask | revisits

            # determine necessary halts
            must_halt = ext_mask.all(-1)
            is_done = is_done | must_halt

            # update the context vector and manual descriptors based on the
             # current route
            route_descs = self.encode_path(nodepair_descs, route,
                                           route_times)
            context = torch.cat((global_feats, route_descs), dim=-1)
            first_iter = False

        # prune extra route sequence padding
        max_route_len = (route > -1).sum(dim=-1).max()
        route = route[:, :max_route_len]

        route_len = (route > -1).sum(-1)

        return route[:, None], logits[:, None], None

    def _get_extension_scores(self, state, context_vec, kv_embeds,
                              key_mask=None):
        # compute scores vs current context and pick an extension path
        # add query "sequence" dimension
        context_vec = context_vec[..., None, :]
        path_chunks = kv_embeds.chunk(2 * self.n_selection_layers + 1, dim=-1)
        keys = path_chunks[::2]
        values = path_chunks[1::2] + (None,)

        # TODO implement multi-head attention
        for query_embedder, lk, lv in zip(self.query_embedders, keys, values):
            query = query_embedder(context_vec)
            scores = torch.bmm(query, lk.transpose(-1, -2)).squeeze(1)
            scores /= self.embed_dim**0.5
            if key_mask is not None:
                scores[key_mask] = TORCH_FMIN
            scores = torch.nn.functional.softmax(scores, dim=-1)
            if lv is not None:
                # the final layer scores are just the selection probabilities,
                 # so don't use them to update the context
                context_vec = torch.bmm(scores[:, None], lv)

        return context_vec.squeeze(1), scores

    def encode_path(self, nodepair_descs, path, path_times):
        if path.ndim == 2:
            added_dim = True
            # add a route dimension
            path = path[:, None]
            path_times = path_times[:, None]
        else:
            added_dim = False
        path_mask = path == -1
        enc = self.path_encoder.encode_nodepairs(nodepair_descs, path)

        path_lens = (~path_mask).sum(-1)
        extra_feats = torch.stack((path_times, path_lens), dim=-1)
        extra_feats = self.path_extras_norm(extra_feats)
        mlp_in = torch.cat((enc, extra_feats), dim=-1)
        path_desc = self.path_mlp(mlp_in)
        if added_dim:
            path_desc.squeeze_(1)
        return path_desc


class NodeWalker(LegacyRouteGeneratorBase):
    def __init__(self, *args, n_heads=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_node_scorer = get_mlp(3, self.embed_dim * 2,
                                         in_dim=self.embed_dim, out_dim=1)
        self.next_node_scorer = get_mlp(3, self.embed_dim * 2, out_dim=1)
        self.next_node_attn = nn.MultiheadAttention(
            self.embed_dim, n_heads, self.dropout, batch_first=True)
        halt_desc = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.register_parameter(name="halt_desc", param=halt_desc)

    def _get_routes(self, n_routes, state, node_descs, extra_nodepair_feats,
                    greedy):
        """
        n_routes: int
        node_descs: batch_size x n_nodes x node_feat_dim
        state: batch_size x n_nodes x n_nodes x state_dim
        greedy: bool
        returns: batch_size x n_routes x route_len, all_logits, None
        """
        log.debug("get routes")
        # doesn't support chunk sizes greater than 1
        assert n_routes == 1

        # fold node descriptors
        node_descs, _ = self.fold_node_descs(node_descs, state)

        # pick first node
        node_scores = self.start_node_scorer(node_descs).squeeze(-1)
        cur_node, logits = select(node_scores, not greedy)
        route = cur_node
        cur_node = cur_node.squeeze(-1)

        # set up loop-tracking variables and other needed values
        is_done = torch.zeros(state.batch_size, device=state.device).bool()
        batch_idxs = state.batch_indices
        node_on_route = torch.zeros((state.batch_size, state.max_n_nodes + 1),
                                    device=state.device).bool()
        adj_mat = state.graph_data.street_adj
        is_adj = adj_mat.isfinite() & (adj_mat > 0)
        halt_desc = self.halt_desc.expand(state.batch_size, -1, -1)

        # assemble a route!
        while not is_done.all():
            # compute scores vs current context and pick an extension path
            # assemble query vector: adjacent nodes and the halt option
            is_valid = is_adj[batch_idxs, cur_node] & ~node_on_route[..., :-1]
            is_valid = is_valid & ~is_done[:, None]
            valid_idxs = get_indices_from_mask(is_valid, dim=-1)
            cdt_feats = node_descs[batch_idxs[:, None], valid_idxs]
            if route.shape[-1] > 1:
                # don't enable halt option unless we have at least two stops
                cdt_feats = torch.cat((cdt_feats, halt_desc), dim=-2)
                valid_idxs = nn.functional.pad(valid_idxs, (0, 1), value=True)

            # select a node
            route_feats = node_descs[batch_idxs[:, None], route]
            route_pad_mask = route == -1
            cdt_descs, _ = self.next_node_attn(
                cdt_feats, route_feats, route_feats,
                key_padding_mask=route_pad_mask, need_weights=False)
            cdt_descs = torch.cat((cdt_descs, cdt_feats), dim=-1)
            scores = self.next_node_scorer(cdt_descs).squeeze(-1)
            scores = get_update_at_mask(scores, valid_idxs == -1, TORCH_FMIN)
            cur_node, next_logit = select(scores, not greedy)
            cur_node = cur_node.squeeze(-1)

            # update route, node_on_route, logits, is_done
            if route.shape[-1] > 1:
                chose_halt = cur_node == cdt_feats.shape[-2] - 1
                is_done = is_done | chose_halt
                cur_node = get_update_at_mask(cur_node, chose_halt, -1)
                next_logit[chose_halt] = 0.0

            route = torch.cat((route, cur_node[:, None]), dim=1)
            logits += next_logit
            node_on_route[batch_idxs, cur_node] = True

        return route[:, None], logits, None


class LatentAttentionEncoder(nn.Module):
    def __init__(self, embed_dim, latent_size=None, n_heads=8, n_layers=2,
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        # the latent embedding.  We don't learn it, but we make it a parameter
         # so that it will be included in the module's state_dict.
        if latent_size is not None:
            latent = nn.Parameter(torch.randn((1, latent_size, embed_dim)))
            self.register_parameter(name="latent", param=latent)
            # self.latent.requires_grad_(False)
        else:
            self.latent = None

        self.nonlin = DEFAULT_NONLIN()
        self.attention_layers = nn.ModuleList(
            nn.MultiheadAttention(embed_dim, n_heads, dropout,
                                  batch_first=True)
            for _ in range(n_layers)
        )

    def forward(self, seq_to_encode, query=None, padding_mask=None,
                embed_pos=False, seqlen_scale=None, residual=False):
        """
        seq_to_encode: batch_size x sequence length x embed dim tensor of the
            sequence(s) to encode
        padding_mask: batch_size x sequence length binary tensor indicating
            which elements of seq_to_encode are padding elements
        embed_pos: if True, sinusoidal position embeddings will be added to the
            sequence before encoding it.
        seqlen_scale: if provided, scale the encoding proportionally to the
            number of nodes, making the latent encoding a weighted sum rather
            than a weighted average; the value of seqlen_scale is the
            proportionality constant.
        residual: if True, the output of each layer is added to the query to
            get the query to the next layer; if False, the output alone is used
            as the query.
        """
        if seq_to_encode is None:
            # can't encode an empty sequence, so the "encoding" is just the
            # latent vector.
            return self.latent

        if seq_to_encode.ndim == 2:
            # there's no batch dimension, so add one
            seq_to_encode = seq_to_encode[None]

        if self.latent is None:
            assert query is not None, \
                "query_base must be provided if there is no latent vector!"

        dev = seq_to_encode.device
        batch_size = seq_to_encode.shape[0]
        if self.latent is not None and query is None:
            batch_latent = self.latent.tile(batch_size, 1, 1).to(device=dev)
            if seq_to_encode.shape[1] == 0:
                # can't encode an empty sequence, so the "encoding" is just the
                # latent vector.
                return batch_latent
            encoding = batch_latent
        else:
            encoding = query

        if embed_pos:
            sinseq = get_sinusoid_pos_embeddings(seq_to_encode.shape[1],
                                                 seq_to_encode.shape[2])
            if seq_to_encode.ndim == 3:
                sinseq = sinseq[None]
            seq_to_encode = seq_to_encode + sinseq.to(device=dev)

        # ignore empty sequences
        if padding_mask is not None:
            seq_is_empty = padding_mask.all(dim=1)
            padding_mask = padding_mask[~seq_is_empty]
            encoding = encoding[~seq_is_empty]
            seq_to_encode = seq_to_encode[~seq_is_empty]
            seq_lens = padding_mask.shape[-1] - padding_mask.sum()
        else:
            seq_lens = torch.ones((batch_size, 1), device=dev)
            seq_lens *= seq_to_encode.shape[-2]

        # encode the input sequences via attention
        for attn_layer in self.attention_layers:
            attn_vec, _ = attn_layer(encoding, seq_to_encode, seq_to_encode,
                                     key_padding_mask=padding_mask)
            if seqlen_scale:
                attn_vec *= seqlen_scale * seq_lens

            if residual:
                encoding = encoding + attn_vec
            else:
                encoding = attn_vec

            if attn_layer is not self.attention_layers[-1]:
                # apply a non-linearity in between attention layers
                encoding = self.nonlin(encoding)

        if padding_mask is not None and seq_is_empty.any():
            batch_latent[~seq_is_empty] = encoding
            encoding = batch_latent

        return encoding


class EdgeGraphNetLayer(MessagePassing):
    def __init__(self, in_node_dim, in_edge_dim=0,
                 out_dim=None, hidden_dim=None, bias=True,
                 nonlin_type=DEFAULT_NONLIN, n_edge_layers=1,
                 dropout=MLP_DEFAULT_DROPOUT, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        if hidden_dim is None:
            hidden_dim = in_node_dim
        if out_dim is None:
            out_dim = in_node_dim
        in_dim = in_node_dim * 2 + in_edge_dim
        layers = [
            get_mlp(n_edge_layers, hidden_dim, nonlin_type, dropout=dropout,
                    bias=bias, in_dim=in_dim, out_dim=out_dim),
        ]
        if n_edge_layers == 1:
            layers.append(nonlin_type())
        self.edge_embedder = nn.Sequential(*layers)

    def forward(self, node_features, edge_index, edge_features):
        if edge_features is not None and edge_features.ndim == 1:
            # edge features have one channel, so ensure there's a dim for them
            edge_features = edge_features[:, None]
        edge_features = self.edge_updater(edge_index, x=node_features,
                                          edge_attr=edge_features)
        size = (node_features.shape[0], node_features.shape[0])
        node_features = self.propagate(edge_index, edge_attr=edge_features,
                                       size=size)

        return node_features, edge_features

    def message(self, edge_attr):
        return edge_attr

    def edge_update(self, x_i, x_j, edge_attr):
        in_vec = torch.cat((x_i, x_j, edge_attr), dim=-1)
        embed = self.edge_embedder(in_vec)
        return embed


class NoOpGraphNet(GraphNetBase):
    def __init__(self, return_edges=False, *args, **kwargs):
        self.return_edges = return_edges
        super().__init__(0, 0, **kwargs)

    def forward(self, data):
        if self.gives_edge_features:
            return data.x, data.edge_attr
        else:
            return data.x

    @property
    def gives_edge_features(self):
        return self.return_edges


class WeightedEdgesNetBase(GraphNetBase):
    def __init__(self, in_edge_dim=1, *args, **kwargs):
        super().__init__(in_edge_dim=1, *args, **kwargs)
        if in_edge_dim > 1:
            # project to range 0 to 1
            self.edge_in_proj = nn.Sequential(
                nn.Linear(in_edge_dim, 1),
                nn.Sigmoid()
            )

    @property
    def gives_edge_features(self):
        return False


class SimplifiedGcn(WeightedEdgesNetBase):
    def __init__(self, n_layers, in_node_dim, out_dim, *args, **kwargs):
        super().__init__(n_layers=n_layers, embed_dim=out_dim,
                         in_node_dim=in_node_dim, out_dim=out_dim,
                         recurrent=False, residual=False, dense=False,
                         *args, **kwargs)
        self.net = SGConv(in_node_dim, out_dim, n_layers)

    def _forward_helper(self, data):
        act1 = self.net(data.x, data.edge_index, data.edge_attr)
        act2 = self.nonlin(act1)
        act3 = self.dropout(act2)
        return act3


class Gcn(WeightedEdgesNetBase):
    def __init__(self, layer_type=GCNConv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.recurrent:
            layers = [layer_type(self.embed_dim, self.embed_dim,
                                 **self.layer_kwargs)] * self.n_layers
        else:
            in_dims = self._get_layer_node_indims()
            layers = [layer_type(in_dim, self.embed_dim, **self.layer_kwargs)
                      for in_dim in in_dims]

        self.layers = nn.ModuleList(layers)


class EdgeGraphNet(GraphNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        make_layer = lambda ind, ied: \
            EdgeGraphNetLayer(ind, ied, out_dim=self.embed_dim,
                              nonlin_type=self.nonlin_type,
                              dropout=self.dropout_rate, **self.layer_kwargs)

        if self.recurrent:
            layers = [make_layer(self.embed_dim, self.embed_dim)] * \
                self.n_layers
        else:
            in_node_dims = self._get_layer_node_indims()
            in_edge_dims = self._get_layer_edge_indims()
            in_edge_dims[0] = self.in_edge_dim
            layers = [make_layer(ind, ied) for ind, ied in
                      zip(in_node_dims, in_edge_dims)]

        self.layers = nn.ModuleList(layers)


    @property
    def gives_edge_features(self):
        return True


def update_connections_matrix(is_linked, new_routes, symmetric_routes=False):
    """Assumes the presence of a batch dimension."""
    batch_idxs = torch.arange(len(new_routes))
    # add dummy padding
    is_linked = torch.nn.functional.pad(is_linked, (0, 1, 0, 1))

    # set newly-connected nodes to be connected
    for stop_idx in range(new_routes.shape[-1] - 1):
        cur_stop = new_routes[:, stop_idx]
        next_stop = new_routes[:, stop_idx+1]
        is_linked[batch_idxs, cur_stop, next_stop] = True

    # re-establish false along the dummy areas
    is_linked[:, -1, :] = False
    is_linked[:, :, -1] = False

    # # is_linked = is_linked[:, :-1, :-1]
    # if symmetric_routes:
    #     # make the matrix symmetric
    #     is_linked = is_linked | is_linked.transpose(-1, -2)

    # edge_costs = torch.zeros_like(is_linked, dtype=torch.float32)
    # edge_costs[~is_linked] = float('inf')
    # edge_costs[is_linked] = 1
    # # edge_costs = edge_costs[:, :-1, :-1]
    # node_idxs = torch.arange(edge_costs.shape[-1], device=edge_costs.device)
    # edge_costs[:, node_idxs, node_idxs] = 0
    # _, _, dists = floyd_warshall(edge_costs, True)
    # # return dists.isfinite()
    # fw_is_linked = dists.isfinite()

    for stop_idx in range(new_routes.shape[-1] - 1):
        cur_stop = new_routes[:, stop_idx]
        next_stop = new_routes[:, stop_idx+1]
        linked_to_cur = is_linked[batch_idxs, :, cur_stop]
        must_be_updated_idxs = torch.where(linked_to_cur)
        update_values = is_linked[must_be_updated_idxs[0],
                                  next_stop[must_be_updated_idxs[0]]]
        is_linked[must_be_updated_idxs[0], must_be_updated_idxs[1]] |= \
            update_values

    # remove the dummy padding
    is_linked = is_linked[:, :-1, :-1]

    if symmetric_routes:
        # make the matrix symmetric
        is_linked = is_linked | is_linked.transpose(-1, -2)

    # finally, return the connections matrix
    return is_linked


def mean_pool_sequence(sequences, padding_mask=None):
    # 2nd-to-last dim is sequence dim, last dim is feature dim
    if padding_mask is not None:
        sequences = sequences * ~padding_mask[..., None]
        seq_lens = (~padding_mask).sum(dim=-1)[..., None]
    else:
        seq_lens = sequences.shape[-2]
    # avoid generating NaNs by dividing by 0
    seq_lens[seq_lens == 0] = 1
    sums = sequences.sum(dim=-2)
    means = sums / seq_lens
    return means


def get_sinusoid_pos_embeddings(seqlen, ndims, posenc_min_rate=1/10000):
    angle_rate_exps = torch.linspace(0, 1, ndims // 2)
    angle_rates = posenc_min_rate ** angle_rate_exps
    positions = torch.arange(seqlen)
    angles_rad = positions[:, None] * angle_rates[None, :]
    sines = torch.sin(angles_rad)
    cosines = torch.cos(angles_rad)
    return torch.cat((sines, cosines), dim=-1)


def assemble_batch_routes_from_tensors(tensors):
    """
    Takes a list of batch_size x n_routes_in_tensor x max_n_stops_in_tensor
        tensors.  Assembles them into a single tensor of shape
        batch_size x total_n_routes x max_n_stops_across_tensors.
    """
    if tensors is None or len(tensors) == 0 or tensors[0] is None:
        # an "empty" input, so return None
        return None
    batch_size = tensors[0].shape[0]
    dev = tensors[0].device
    max_route_len = max([tt.shape[-1] for tt in tensors])
    n_routes = sum([tt.shape[-2] for tt in tensors])
    out = torch.zeros((batch_size, n_routes, max_route_len), device=dev)
    routes_so_far = 0
    for tt in tensors:
        end_idx = routes_so_far + tt.shape[-2]
        out[:, routes_so_far:end_idx, :tt.shape[-1]] = tt
        routes_so_far = end_idx
    return out


def get_indices_from_mask(mask, dim=0, pad_value=-1):
    """Given a boolean mask, return the a tensor of indices of the True
       elements."""
    n_indices = mask.sum(dim=dim)
    shape = list(mask.shape)
    shape[dim] = n_indices.max().item()
    dev = mask.device
    indices = torch.full(shape, pad_value, dtype=int, device=dev)

    dims = [1] * len(shape)
    dims[dim] = -1
    # add a 1-length dim for each other dimension of the mask
    indices_source = torch.arange(mask.shape[dim], device=dev).reshape(*dims)
    indices_source = indices_source.expand_as(mask)

    var_slice_mask = get_variable_slice_mask(indices, dim, tos=n_indices)
    indices[var_slice_mask] = indices_source[mask]

    return indices


def expand_batch_dim(tnsr_to_expand, batch_size, dim=0):
    if tnsr_to_expand.shape[dim] != 1:
        # there's no unitary dimension at the start, so add one
        tnsr_to_expand = tnsr_to_expand.unsqueeze(dim)

    repeats = [1] * tnsr_to_expand.dim()
    repeats[dim] = batch_size
    expanded = tnsr_to_expand.repeat(repeats)
    return expanded


def format_stat_msg(tensor, name=''):
    tensor = tensor.to(dtype=torch.float32)
    msg = f"{name} mean: {tensor.mean().item():.2f} " \
          f"std: {tensor.std().item():.2f} " \
          f"min: {tensor.min().item():.2f} " \
          f"max: {tensor.max().item():.2f} " \
          f"size: {tensor.numel()}"
    return msg
