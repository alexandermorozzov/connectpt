"""Atomic recovery at completed CPU construction updates, separate from actor exports."""
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..policy.models import FeatureNorm


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class RecordedSampler:
    """Retain the current random permutation and cursor without replaying augmentation.

    Materializing RandomSampler consumes the same global seed as its first next();
    all subsequent permutation work uses the sampler's local generator.
    """
    def __init__(self, sampler):
        self.source = sampler
        self.order = None
        self.position = 0

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        if self.order is None:
            self.order, self.position = list(self.source), 0
        while self.position < len(self.order):
            index = self.order[self.position]
            self.position += 1
            yield index
        self.order, self.position = None, 0

    def state_dict(self):
        return dict(order=self.order, position=self.position)

    def load_state_dict(self, state):
        self.order, self.position = state['order'], state['position']


def module_state(model):
    return dict(weights=model.state_dict(), modes={n: m.training for n, m in model.named_modules()},
                pending_norm={n: dict(new_mean=m.new_mean, new_var=m.new_var, new_count=m.new_count)
                              for n, m in model.named_modules() if isinstance(m, FeatureNorm)})


def load_module(model, state):
    model.load_state_dict(state['weights'], strict=True)
    for name, module in model.named_modules():
        module.training = state['modes'][name]
        if isinstance(module, FeatureNorm):
            # update() promotes the initial integer counter to a floating average.
            # load_state_dict would cast it back into the fresh integer buffer.
            key = f'{name}.avg_count' if name else 'avg_count'
            module.avg_count = state['weights'][key].clone()
        for key, value in state['pending_norm'].get(name, {}).items():
            setattr(module, key, value)


class TrainingRecovery:
    """Opt-in exact same-runtime recovery for the runner's CPU/file construction jobs.

    Saves no dataset graphs except the currently active environment batch. Other
    training paths retain their existing loader and checkpoint behavior.
    """
    format = 'connectpt.training_state.v1'
    data_fields = ('finished', 'route_count', 'graph_instances_loaded', 'graph_batches_loaded', 'episode_statistics')
    env_fields = ('state', 'result', 'params', 'steps', 'single_route', 'ended')
    critic_fields = ('_ret_mean', '_ret_var', '_ret_std', '_ret_initialized')
    validation_fields = ('last_monitor_step', 'validated_cost', 'prefix_best')

    def __init__(self, path, identity, *, milestone_period=200):
        self.path = Path(path)
        self.identity = identity
        self.milestone_period = milestone_period
        self.sampler = None

    def prepare(self, data):
        loader = data.train_loader
        if data.online or loader.num_workers or loader.generator is not None:
            raise ValueError('Exact recovery requires a file construction dataset and a synchronous loader')
        self.sampler = RecordedSampler(loader.sampler)
        data.train_loader = DataLoader(loader.dataset, batch_size=loader.batch_size,
            sampler=self.sampler, drop_last=loader.drop_last, num_workers=0,
            pin_memory=loader.pin_memory)

    def save(self, *, env, policy, critic, optimizer, data, tracker, logger_list,
             iteration, phase, stats, elapsed):
        if next(policy.parameters()).device.type != 'cpu' or critic.params.shared:
            raise ValueError('Training recovery currently supports CPU construction with a separate critic')
        if critic._curr_estimate is not None:
            raise RuntimeError('Cannot save recovery state in the middle of a critic update')
        best = module_state(tracker.best_policy) if tracker.best_policy is not None else None
        payload = dict(format=self.format, identity=self.identity, iteration=iteration, phase=phase,
            actor=module_state(policy), critic=module_state(critic.model),
            optimizer=optimizer.state_dict(), critic_optimizer=critic.optim.state_dict(),
            critic_runtime={k: getattr(critic, k) for k in self.critic_fields},
            environment={k: getattr(env, k) for k in self.env_fields if hasattr(env, k)},
            data={k: getattr(data, k) for k in self.data_fields}, sampler=self.sampler.state_dict(),
            extra=getattr(data, 'checkpoint_extra', {}),
            validation={k: getattr(tracker.validation, k) for k in self.validation_fields
                        if hasattr(tracker.validation, k)},
            tracker=dict(best_cost=tracker.best_cost, best_policy=best,
                         last_step=tracker.last_step, last_cost=tracker.last_cost,
                         monitor_final=tracker.monitor_final, final_monitor_step=tracker.final_monitor_step),
            loggers=[dict(rows=logger.rows, columns=logger.columns) if hasattr(logger, 'rows') else None
                     for logger in logger_list], stats=stats, elapsed=elapsed, rng=rng_state())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + f'.{os.getpid()}.tmp')
        try:
            torch.save(payload, temporary)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)
        if phase == 'monitor' and iteration and iteration % self.milestone_period == 0:
            milestone = self.path.with_name(f'training_at_{iteration}.pt')
            temporary = milestone.with_name(milestone.name + '.tmp')
            # Copy the complete payload, not actor-only weights.
            import shutil
            shutil.copyfile(self.path, temporary)
            os.replace(temporary, milestone)

    def load(self, *, env, policy, critic, optimizer, data, tracker, logger_list):
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        if payload.get('format') != self.format or payload.get('identity') != self.identity:
            raise ValueError('Recovery checkpoint format or training configuration does not match')
        load_module(policy, payload['actor'])
        load_module(critic.model, payload['critic'])
        optimizer.load_state_dict(payload['optimizer'])
        critic.optim.load_state_dict(payload['critic_optimizer'])
        for key, value in payload['critic_runtime'].items():
            setattr(critic, key, value)
        for key, value in payload['environment'].items():
            setattr(env, key, value)
        for key, value in payload['data'].items():
            setattr(data, key, value)
        self.sampler.load_state_dict(payload['sampler'])
        # Recreating an iterator consumes a base seed; restore the saved RNG last.
        data.initialize()
        for key, value in payload['extra'].items():
            data.checkpoint_extra[key].copy_(value)
        for key, value in payload['validation'].items():
            setattr(tracker.validation, key, value)
        for key in ('best_cost', 'last_step', 'last_cost', 'monitor_final', 'final_monitor_step'):
            setattr(tracker, key, payload['tracker'][key])
        if payload['tracker']['best_policy'] is not None:
            tracker.best_policy = copy.deepcopy(policy)
            load_module(tracker.best_policy, payload['tracker']['best_policy'])
        for logger, state in zip(logger_list, payload['loggers']):
            if state is not None:
                logger.rows, logger.columns = state['rows'], state['columns']
            elif hasattr(logger, 'resume'):
                logger.resume(payload['iteration'])
        restore_rng(payload['rng'])
        return payload
