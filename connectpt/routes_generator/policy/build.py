"""Build and load a policy from a fully resolved, typed profile."""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ..env.features import FeatureParams, dims
from ..common.parameters import CostConstants
from ..policy.checkpoints import load_model_weights
from .models import GraphAttnNet, PathCombiningRouteGenerator, TrimPathCombiningRouteGenerator


@dataclass(frozen=True)
class CommonParams:
    dropout: float
    nonlin_type: str
    embed_dim: int


@dataclass(frozen=True)
class DecoderParams:
    n_nodepair_layers: int
    n_pathscorer_layers: int
    pathscorer_hidden_dim: int
    n_halt_layers: int
    n_halt_heads: int
    halt_scorer_type: str
    serial_halting: bool
    force_linking_unlinked: bool
    logit_clip: float | None
    max_act_len: int | None
    feature_norm_unbiased: bool
    use_redundancy_features: bool
    temperature: float
    fixed_frequency_Hz: float
    only_routes_with_demand_are_valid: bool


@dataclass(frozen=True)
class TrimDecoderParams(DecoderParams):
    n_trim_scorer_layers: int
    trim_scorer_hidden_dim: int
    forbid_halt_when_overlong: bool
    allow_trim_below_min: bool


@dataclass(frozen=True)
class BackboneParams:
    n_layers: int
    n_heads: int
    dense: bool
    use_norm: bool
    recurrent: bool
    residual: bool
    n_proj_layers: int
    out_dim: int | None
    layer_kwargs: dict


@dataclass(frozen=True)
class ModelParams:
    common: CommonParams
    decoder_type: str
    decoder: DecoderParams
    backbone_type: str
    backbone: BackboneParams
    features: FeatureParams

    @classmethod
    def from_values(cls, values):
        generator = values["route_generator"]
        decoder_cls = DECODER_PARAMS[generator["type"]]
        return cls(
            CommonParams(**values["common"]),
            generator["type"],
            decoder_cls(**generator["kwargs"]),
            values["backbone_gn"]["net_type"],
            BackboneParams(**values["backbone_gn"]["kwargs"]),
            FeatureParams.from_values(values["features"]),
        )


@dataclass(frozen=True)
class ModelProfileParams:
    config: str
    checkpoint_path: str | None


@dataclass(frozen=True)
class PolicyProfile:
    model: ModelParams
    constants: CostConstants
    objectives: tuple[str, ...]
    low_memory_mode: bool
    checkpoint_path: Path | None
    device: torch.device


POLICIES = {
    "PathCombiningRouteGenerator": PathCombiningRouteGenerator,
    "TrimPathCombiningRouteGenerator": TrimPathCombiningRouteGenerator,
}
DECODER_PARAMS = {"PathCombiningRouteGenerator": DecoderParams, "TrimPathCombiningRouteGenerator": TrimDecoderParams}
BACKBONES = {"graph attn": GraphAttnNet}


def load_policy(profile: PolicyProfile):
    params = profile.model
    sizes = dims(params.features)
    backbone = BACKBONES[params.backbone_type](
        **asdict(params.common), **asdict(params.backbone), in_node_dim=sizes["node"], in_edge_dim=sizes["pair"]
    )
    policy = POLICIES[params.decoder_type](
        backbone_net=backbone,
        mean_stop_time_s=profile.constants.mean_stop_time_s,
        symmetric_routes=profile.constants.symmetric_routes,
        low_memory_mode=profile.low_memory_mode,
        feature_set=params.features,
        **asdict(params.common),
        **asdict(params.decoder),
    )
    policy.objectives = profile.objectives
    policy.to(profile.device)
    if profile.checkpoint_path is not None:
        load_model_weights(policy, profile.checkpoint_path, strict=True, map_location=profile.device)
    return policy
