"""Single loader/saver for route-model checkpoints.

The hard invariant of the whole refactor: trained weights must keep loading
*directly* into the existing model classes
(:class:`PathCombiningRouteGenerator` / :class:`TrimPathCombiningRouteGenerator`)
through ``strict=True``. Adapters and run-wrappers never own the ``state_dict``;
they delegate to this store.

Two on-disk formats are supported, both readable by the same loader:

* **legacy** -- ``torch.save(model.state_dict(), path)`` (a bare state dict);
* **v2**     -- ``torch.save({"format_version": 2, "state_dict": ..., ...})``
  with optional ``config`` / ``model_class`` metadata.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:  # OmegaConf is a hard dep of the cfg layer, but keep saving usable without it
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - omegaconf is always installed in practice
    OmegaConf = None


class CheckpointStore:
    """Load/save model weights in a format-agnostic way."""

    @staticmethod
    def extract_state_dict(obj: Any) -> dict[str, torch.Tensor]:
        """Return the bare ``state_dict`` from either a legacy or v2 payload."""
        if isinstance(obj, dict) and "state_dict" in obj:
            return obj["state_dict"]
        return obj

    @staticmethod
    def load_model_weights(
        model,
        path: str | Path,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ):
        """Load weights straight into ``model`` (no wrapper, no key rewriting)."""
        obj = torch.load(path, map_location=map_location)
        state_dict = CheckpointStore.extract_state_dict(obj)
        return model.load_state_dict(state_dict, strict=strict)

    @staticmethod
    def save_model_weights(
        model,
        path: str | Path,
        cfg=None,
        *,
        format_version: int = 2,
    ) -> None:
        """Save weights in the v2 format (state dict + metadata).

        Legacy readers that expect a bare state dict keep working because
        :meth:`extract_state_dict` understands both layouts.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, Any] = {
            "format_version": format_version,
            "model_class": type(model).__name__,
            "state_dict": model.state_dict(),
        }

        if cfg is not None and OmegaConf is not None:
            payload["config"] = OmegaConf.to_container(cfg, resolve=True)
        elif cfg is not None:
            payload["config"] = cfg

        torch.save(payload, path)

    @staticmethod
    def assert_state_dict_compatible(model, path: str | Path) -> None:
        """Raise if ``model`` and the checkpoint at ``path`` disagree on keys.

        Used by the regression tests to catch the moment a refactor renames a
        ``state_dict`` key (the failure mode that would silently break
        ``strict=True`` loading of old checkpoints).
        """
        obj = torch.load(path, map_location="cpu")
        state_dict = CheckpointStore.extract_state_dict(obj)

        model_keys = set(model.state_dict())
        ckpt_keys = set(state_dict)

        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        assert not missing, f"keys in model but not checkpoint: {sorted(missing)[:10]}"
        assert not unexpected, f"keys in checkpoint but not model: {sorted(unexpected)[:10]}"

        model.load_state_dict(state_dict, strict=True)
