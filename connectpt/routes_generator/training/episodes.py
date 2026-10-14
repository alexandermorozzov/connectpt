"""Episode-source hooks, training preferences and owned reward/length counters."""
from abc import ABC,abstractmethod
from dataclasses import dataclass

import torch

from ..objectives.preferences import PreferenceSampler,state_preferences
from .const import PreferenceDistribution


@dataclass(frozen=True)
class EpisodePreferences:
    """Training-episode preferences: distribution and lifetime. Validation keeps the configured sampler.

    ``sample`` consumes the global torch RNG and returns State weight keys -> Tensor[count].
    """
    sampler: PreferenceSampler
    distribution: PreferenceDistribution
    resample_on_terminal: bool

    def sample(self,count,device):
        if self.distribution is PreferenceDistribution.SIMPLEX:
            return state_preferences(self.sampler.sample_simplex_named(count,device))
        return state_preferences(self.sampler.sample_named(count,device))


@dataclass
class EpisodeStatistics:
    rewards: torch.Tensor
    lengths: torch.Tensor

    def record(self,rewards,ended,active):
        self.lengths -= (~active).long()
        self.rewards.add_(rewards)
        self.lengths.add_(1)
        completed_rewards=self.rewards[ended].clone()
        completed_lengths=self.lengths[ended].clone()
        self.rewards[ended]=0
        self.lengths[ended]=0
        return completed_rewards,completed_lengths


class EpisodeSource(ABC):
    def __init__(self):
        self.episode_statistics=None

    def start_episode(self,batch_size,device):
        self.episode_statistics=EpisodeStatistics(torch.zeros(batch_size,device=device),
                                                  torch.zeros(batch_size,dtype=torch.long,device=device))

    def record_step(self,rewards,ended,active):
        if self.episode_statistics is None:
            raise RuntimeError('episode counters require a started batch')
        return self.episode_statistics.record(rewards,ended,active)

    @abstractmethod
    def initialize(self): ...

    @abstractmethod
    def warmup(self,count): ...

    @abstractmethod
    def get(self,progress): ...

    @abstractmethod
    def start(self,env): ...

    @abstractmethod
    def prepare_step(self,env): ...

    @abstractmethod
    def after_step(self,env,done): ...

    @abstractmethod
    def bootstrap_mask(self,env): ...

    @abstractmethod
    def final_evaluation(self): ...
