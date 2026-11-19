"""Opt-in scalar observers. No gradients, trajectory dumps or training RNG draws."""
from dataclasses import dataclass
import torch

from ..common.logging import log
from .records import ScalarDiagnosticRecord
from .checkpoints import preserve_rng


@dataclass(frozen=True)
class ScalarDiagnosticParams:
    reference_gamma: float
    validation_graphs: int
    validation_samples: int
    validation_alphas: tuple
    validation_batch_size: int
    validation_period: int
    seed: int
    selection_alphas: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, 'validation_alphas', tuple(self.validation_alphas))
        object.__setattr__(self, 'selection_alphas', tuple(self.selection_alphas))
        if not 0 < self.reference_gamma <= 1:
            raise ValueError('Diagnostic reference gamma must be in (0,1]')
        if min(self.validation_graphs,self.validation_samples,self.validation_batch_size,self.validation_period)<1:
            raise ValueError('Diagnostic evaluation sizes/period must be positive')
        if any(not 0 <= alpha <= 1 for alpha in self.validation_alphas):
            raise ValueError('Diagnostic alpha outside [0,1]')
        if not set(self.selection_alphas).issubset(self.validation_alphas):
            raise ValueError('Selection preferences must be included in diagnostic validation')


def alpha_groups(alpha):
    """Disjoint bins; endpoints stay separate from intermediate preferences."""
    return {
        'a000': alpha == 0, 'a100': alpha == 1,
        'a000_025': (alpha > 0) & (alpha < .25),
        'a025_050': (alpha >= .25) & (alpha < .5),
        'a050_075': (alpha >= .5) & (alpha < .75),
        'a075_100': (alpha >= .75) & (alpha < 1),
    }


class Means:
    def __init__(self):
        self.values = {}

    def add(self, name, values):
        values = torch.as_tensor(values).detach().double().flatten()
        if values.numel():
            total, count = self.values.get(name, (0., 0))
            self.values[name] = total + float(values.sum()), count + values.numel()

    def result(self):
        return {name: total/count for name, (total, count) in self.values.items() if count}


class ScalarDiagnostics:
    """Observe one ATT/RTT construction run. State is O(batch), checkpointable.

    ``transition`` receives evaluations owned by the collector; reads only.
    ``rollout`` compares pre-update values to observed full discounted tails,
    excludes tails cut by the horizon and uses no GAE/bootstrap targets.
    ``minibatch`` receives detached existing tensors and retains only scalars.
    Validation uses separate environments and preserves Python/NumPy/torch RNG.
    """
    def __init__(self, params, *, gamma, objective_names, env_params):
        if tuple(objective_names) != ('ATT', 'RTT') or not env_params.diff_reward:
            raise ValueError('Scalar diagnostics require ATT/RTT construction with difference rewards')
        self.params, self.gamma, self.env_params = params, gamma, env_params
        self.episodes, self.updates = Means(), Means()
        self.identity_error = 0.
        self.state = None

    def attach(self, data):
        b = data.batch_size
        values = dict(
            batch_id=torch.tensor(-1,dtype=torch.long),
            steps=torch.zeros(b, dtype=torch.long), initial_cost=torch.zeros(b, dtype=torch.float64),
            first_connected=torch.full((b,), -1, dtype=torch.long),
            connected_routes=torch.full((b,), -1, dtype=torch.long),
            G1=torch.zeros((b, 3), dtype=torch.float64),
            Ggamma=torch.zeros((b, 3), dtype=torch.float64),
            Greference=torch.zeros((b, 3), dtype=torch.float64),
        )
        if not hasattr(data, 'checkpoint_extra'):
            data.checkpoint_extra = {}
        for key, value in values.items():
            data.checkpoint_extra['diag_' + key] = value
        self.state = values

    @torch.no_grad()
    def start(self,data):
        # Construction abandons residual repeated episodes when loading a new
        # city batch. They must not be joined to episodes on the next cities.
        s=self.state
        if int(s['batch_id']) == data.graph_batches_loaded:return
        self.episodes.add('truncated_episode_count',torch.tensor([int((s['steps']>0).sum())]))
        for key,value in s.items():
            if key!='batch_id':value.fill_(-1 if key in ('first_connected','connected_routes') else 0)
        s['batch_id'].fill_(data.graph_batches_loaded)

    @torch.no_grad()
    def transition(self, previous, current, active, ended, state):
        s = self.state
        if s is None:
            raise RuntimeError('Attach diagnostic state before collection')
        scale = self.env_params.reward_scale
        components = (previous.goals-current.goals) * current.preferences * scale
        penalties = torch.zeros_like(current.cost)
        for i,weight in enumerate(current.penalty_weights):
            penalties += (previous.penalties[:,i]-current.penalties[:,i])*weight*scale
        rewards = torch.cat((components, penalties[:, None]), -1).double().cpu()
        active, ended = active.cpu(), ended.cpu()
        starting = active & (s['steps'] == 0)
        s['initial_cost'][starting] = previous.cost.detach().double().cpu()[starting]
        rewards *= active[:, None]
        s['G1'] += rewards
        s['Ggamma'] += torch.pow(self.gamma, s['steps'].double())[:, None] * rewards
        s['Greference'] += torch.pow(self.params.reference_gamma, s['steps'].double())[:, None] * rewards
        s['steps'] += active.long()
        connected = active & (s['first_connected'] < 0) & (current.n_disconnected_demand_edges.cpu() == 0)
        s['first_connected'][connected] = s['steps'][connected]
        route_count = state.n_finished_routes.cpu() + state.has_current_route.cpu().long()
        s['connected_routes'][connected] = route_count[connected].long()
        if not ended.any():
            return
        alpha = current.preferences[:, 0].detach().cpu()
        terminal = dict(cost=current.cost.detach().cpu(), ATT=current.mean_demand_time.detach().cpu()/60,
                        RTT=current.total_route_time.detach().cpu()/60,
                        feasible=(~current.are_constraints_violated()).float().cpu())
        error = (s['G1'].sum(-1)-(s['initial_cost']-current.cost.detach().double().cpu())*scale).abs()
        self.identity_error = max(self.identity_error, float(error[ended].max()))
        for group, mask in alpha_groups(alpha).items():
            selected = mask & ended
            if not selected.any():continue
            prefix = group + '/'
            self.episodes.add(prefix+'completed_count', torch.ones(1)*int(selected.sum()))
            for name, value in terminal.items():self.episodes.add(prefix+name, value[selected])
            self.episodes.add(prefix+'steps', s['steps'][selected])
            for name in ['G1', 'Ggamma', 'Greference']:
                self.episodes.add(prefix+name, s[name][selected].sum(-1))
            for i, name in enumerate(['ATT', 'RTT', 'constraints']):
                self.episodes.add(prefix+'Ggamma_'+name, s['Ggamma'][selected, i])
                self.episodes.add(prefix+'Greference_'+name, s['Greference'][selected, i])
            reached = selected & (s['first_connected'] >= 0)
            self.episodes.add(prefix+'connected_fraction', (s['first_connected'][selected] >= 0).float())
            self.episodes.add(prefix+'steps_to_connect', s['first_connected'][reached])
            self.episodes.add(prefix+'routes_to_connect', s['connected_routes'][reached])
        for key, value in s.items():
            if key!='batch_id':value[ended] = -1 if key in ('first_connected', 'connected_routes') else 0

    @torch.no_grad()
    def rollout(self, batch, logger_list, iteration):
        weights = batch.preferences
        rewards = batch.rewards if batch.rewards.ndim == 2 else (batch.rewards*weights).sum(-1)
        values = batch.values if batch.values.ndim == 2 else (batch.values*weights).sum(-1)
        tail = torch.zeros_like(rewards[0]); observed = torch.zeros_like(batch.dones[0])
        targets = torch.zeros_like(rewards); valid = torch.zeros_like(batch.dones)
        for t in reversed(range(len(rewards))):
            tail = rewards[t] + self.gamma * tail * ~batch.dones[t]
            observed = batch.dones[t] | observed
            targets[t], valid[t] = tail, observed
        connected = torch.stack([~((~s.has_path) & (s.demand > 0)).flatten(1).any(-1) for s in batch.states])
        error = values-targets
        measured = Means()
        for group, group_mask in alpha_groups(weights[..., 0]).items():
            for stage, stage_mask in [('before_connect', ~connected), ('after_connect', connected)]:
                mask = group_mask & stage_mask & valid & batch.score
                if not mask.any():continue
                e = error[mask].double();prefix=group+'/'+stage+'/'
                measured.add(prefix+'mc_rmse', e.square().mean().sqrt().reshape(1))
                measured.add(prefix+'mc_bias', e)
                measured.add(prefix+'mc_count', torch.tensor([e.numel()]))
                measured.add(prefix+'mc_target_std', targets[mask].double().std(unbiased=False).reshape(1))
        episode = self.episodes.result()
        # Counts are sums, unlike means of episode statistics.
        for name, (total, _) in self.episodes.values.items():
            if name.endswith('/completed_count'):episode[name]=total
        episode['identity_error_max'] = self.identity_error
        log(logger_list, ScalarDiagnosticRecord(episode=episode, critic=measured.result()), iteration)
        self.episodes, self.identity_error = Means(), 0.

    @torch.no_grad()
    def minibatch(self, ratios, advantages, raw_advantages, preferences, *, epsilon):
        if advantages.ndim == 1:
            clipped = ((advantages > 0) & (ratios > 1+epsilon)) | ((advantages < 0) & (ratios < 1-epsilon))
            clipped = clipped.float();scalar_adv = advantages
        else:
            clipped = ((advantages > 0) & (ratios[:, None] > 1+epsilon)) | ((advantages < 0) & (ratios[:, None] < 1-epsilon))
            clipped = (clipped.float()*preferences).sum(-1);scalar_adv=(advantages*preferences).sum(-1)
            if raw_advantages.shape[0]>1:
                std=raw_advantages.std(0)
                self.updates.add('adv_scale_RTT_over_ATT', (std[1]/(std[0]+1e-8)).reshape(1))
        for group, mask in alpha_groups(preferences[:, 0]).items():
            if not mask.any():continue
            self.updates.add(group+'/active_clip_fraction', clipped[mask])
            self.updates.add(group+'/positive_adv_fraction', (scalar_adv[mask] > 0).float())
            raw=raw_advantages[mask]
            if raw.ndim > 1:raw=(raw*preferences[mask]).sum(-1)
            self.updates.add(group+'/raw_adv_std', raw.std(unbiased=False).reshape(1))

    def flush_update(self, logger_list, iteration):
        log(logger_list, ScalarDiagnosticRecord(update=self.updates.result()), iteration)
        self.updates = Means()

    @torch.no_grad()
    def validation(self, policy, logger_list, step, *, data, cost, env_factory):
        if step % self.params.validation_period:return
        with preserve_rng():
            graphs = [data.full_validation[i] for i in data.validation_indices[:self.params.validation_graphs]]
        if not graphs:return
        modes = [(module, module.training) for module in policy.modules()]
        measured = Means()
        try:
            policy.eval()
            with preserve_rng():
                for ai, alpha in enumerate(self.params.validation_alphas):
                    pref=dict(demand_time_weight=alpha,route_time_weight=1-alpha,median_connectivity_weight=0.)
                    records=[]
                    for mode, draws in [('greedy', 1), ('sample', self.params.validation_samples)]:
                        inputs=[(city,draw) for draw in range(draws) for city in range(len(graphs))]
                        for offset in range(0,len(inputs),self.params.validation_batch_size):
                            slots=inputs[offset:offset+self.params.validation_batch_size]
                            torch.manual_seed(self.params.seed+ai*1000+offset)
                            env=env_factory(cost.with_params(**pref),policy.feature_set,self.env_params,policy.action_spec,node_normalizer=policy.node_norm)
                            env.reset([graphs[city].clone() for city,_ in slots],pref)
                            g=torch.zeros(len(slots));reference=torch.zeros(len(slots));t=0
                            while not env.is_done().all():
                                obs,mask=env.observe();active=~env.is_done().clone()
                                output=policy.act(obs,mask,greedy=mode=='greedy')
                                previous=env.result.cost.detach().clone();env.step(output.action)
                                reward=(previous-env.result.cost)*active*self.env_params.reward_scale
                                g+=self.gamma**t*reward;reference+=self.params.reference_gamma**t*reward;t+=1
                            result=env.result;prefix=f'a{round(alpha*100):03d}/{mode}/'
                            for name,value in [('cost',result.cost),('ATT',result.mean_demand_time/60),('RTT',result.total_route_time/60),
                                ('feasible_fraction',(~result.are_constraints_violated()).float()),('Ggamma',g),('Greference',reference)]:
                                measured.add(prefix+name,value)
                            if mode=='sample':
                                records.extend((city,float(result.cost[j]),float(g[j]),float(reference[j])) for j,(city,_) in enumerate(slots))
                    for city in range(len(graphs)):
                        rows=[r for r in records if r[0]==city]
                        c=torch.tensor([r[1] for r in rows],dtype=torch.float64)
                        for name,index in [('current',2),('reference',3)]:
                            ret=torch.tensor([r[index] for r in rows],dtype=torch.float64)
                            dc=c[:,None]-c;dg=ret[:,None]-ret
                            pairs=torch.triu(torch.ones_like(dc,dtype=torch.bool),diagonal=1)&(dc.abs()>1e-6)&(dg.abs()>1e-6)
                            prefix=f'a{round(alpha*100):03d}/alignment_{name}/'
                            measured.add(prefix+'pair_count', pairs.sum().reshape(1))
                            if pairs.any():measured.add(prefix+'rank_reversal_fraction', ((dc*dg>0)[pairs]).double())
                            regret=100*(c[ret.argmax()]-c.min())/c.min().clamp_min(1e-8)
                            measured.add(prefix+'final_cost_regret_percent',regret.reshape(1))
        finally:
            for module,mode in modes:module.train(mode)
        values=measured.result()
        balanced=None
        if self.params.selection_alphas:
            balanced=sum(values[f'a{round(alpha*100):03d}/sample/cost'] for alpha in self.params.selection_alphas)/len(self.params.selection_alphas)
            values['balanced_sample_cost']=balanced
        log(logger_list,ScalarDiagnosticRecord(validation=values),step)
        return balanced
