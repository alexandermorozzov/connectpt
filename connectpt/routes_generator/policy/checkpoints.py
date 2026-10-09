"""Read legacy/v2 model weights directly; save metadata as a plain dictionary."""
from pathlib import Path
from typing import Any
import torch


def extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
    """Return the bare ``state_dict`` from either a legacy or v2 payload."""
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def load_model_weights(
    model,
    path: str | Path,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
):
    """Load weights straight into ``model`` (no wrapper, no key rewriting)."""
    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict) and "feature_names" in obj:
        actual = model.feature_set.metadata()
        if obj["feature_names"] != actual["feature_names"] or obj["dims"] != actual["dims"]:
            raise ValueError(
                f"checkpoint features {obj['feature_names']} differ from policy features {actual['feature_names']}"
            )
    state_dict = extract_state_dict(obj)
    return model.load_state_dict(state_dict, strict=strict)


def save_model_weights(
    model,
    path: str | Path,
    config: dict[str, Any] | None = None,
    *,
    format_version: int = 2,
) -> None:
    """Save weights in the v2 format (state dict + metadata).

    Readers can use :func:`extract_state_dict` for both payload layouts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format_version": format_version,
        "model_class": type(model).__name__,
        "state_dict": model.state_dict(),
    }
    payload.update(model.feature_set.metadata())
    payload["objectives"] = list(model.objectives)

    if config is not None:
        if not isinstance(config, dict):
            raise TypeError("checkpoint config must be a plain dict")
        payload["config"] = config

    torch.save(payload, path)
