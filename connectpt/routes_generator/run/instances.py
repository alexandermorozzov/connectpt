"""Resolve instance configuration and assemble physical city data with routes."""
from dataclasses import replace
from pathlib import Path

from omegaconf import OmegaConf

from ..city.instances import Instance, InstanceParams, load_city
from ..network.routes import read_routes_txt
from .const import INSTANCE_CONFIG_DIR
from .context import resolve_under_root
from .schemas import params


def instance_spec(name: str):
    """Read the named instance group at the configuration boundary."""
    path = INSTANCE_CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no spec for instance {name!r}: expected {path} with route bounds")
    return OmegaConf.load(path)


def instance_params(instance, *, instances_dir=None) -> InstanceParams:
    if isinstance(instance, (str, Path)):
        path = resolve_under_root(instance)
        if path.is_dir():
            instances_dir, instance = path.parent, path.name
        instance = params(InstanceParams, instance_spec(str(instance)))
    if not isinstance(instance, InstanceParams):
        raise TypeError("instance must be a name, directory or InstanceParams")
    folder = Path(instances_dir) / instance.name if instances_dir is not None else instance.path
    return replace(instance, path=resolve_under_root(folder))


def load_instance(instance, *, instances_dir=None, require_routes=True) -> Instance:
    """Load a configured city and optionally validate its initial route network."""
    resolved = instance_params(instance, instances_dir=instances_dir)
    city = load_city(resolved)
    routes = read_routes_txt(resolved.path / "Routes.txt", n_nodes=city.tensors["node_locs"].shape[0], spec=city.spec) if require_routes else None
    return replace(city, init_routes=routes)
