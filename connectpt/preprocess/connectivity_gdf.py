from collections.abc import Callable
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from connectpt.routes_generator.citygraph_dataset import get_dataset_from_config
from connectpt.routes_generator.core.paths import resolve_under_root
from connectpt.routes_generator.data import as_route_tensor, load_matrices
from connectpt.routes_generator.objectives import CostFactory
from connectpt.routes_generator.transit_time_estimator import RouteGenBatchState


def _bind(gdf: gpd.GeoDataFrame, xy: np.ndarray) -> pd.Series:
    """Match node coordinates to intersecting polygons, keeping one node per polygon."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame")
    if not isinstance(xy, np.ndarray):
        raise TypeError("xy must be a NumPy array")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (n_nodes, 2)")

    points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=gdf.crs)
    hits = gpd.sjoin(points, gdf[["geometry"]], predicate="intersects")
    centroids = gdf.centroid.loc[hits["index_right"]].values
    hits["dist"] = hits.distance(gpd.GeoSeries(centroids, index=hits.index))
    hits = hits.sort_values("dist").reset_index(names="node")
    hits = hits.drop_duplicates("node").drop_duplicates("index_right")
    return hits.set_index("index_right")["node"].sort_index()


def _matrix(routes: torch.Tensor, tensors: dict) -> np.ndarray:
    """Compute pairwise connectivity travel times in minutes for one route network."""
    if not isinstance(routes, torch.Tensor):
        raise TypeError("routes must be a torch.Tensor")
    if routes.ndim != 3 or routes.shape[0] != 1:
        raise ValueError("routes must have shape (1, n_routes, max_route_len)")
    if routes.shape[1] == 0 or routes.shape[2] == 0:
        raise ValueError("routes must contain at least one nonempty route")
    if not isinstance(tensors, dict) or not all(isinstance(value, torch.Tensor) for value in tensors.values()):
        raise TypeError("tensors must be a dictionary of torch.Tensor values")

    cost_obj = CostFactory.build_cost()
    data = next(iter(DataLoader(get_dataset_from_config(
        OmegaConf.create({"type": "tensor"}), tensors=tensors), batch_size=1)))
    lens = (routes[0] >= 0).sum(-1)
    state = RouteGenBatchState(data, cost_obj, routes.shape[1], int(lens.min()), int(lens.max()))
    state.add_new_routes(routes)
    return (cost_obj.connectivity_pair_times(state)[0] / 60.0).cpu().numpy()


def map_polygons(gdf: gpd.GeoDataFrame, instance_dir: str | Path) -> gpd.GeoDataFrame:
    """Return polygons with matched instance node IDs in the ``node`` column."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("gdf must be a GeoDataFrame")
    if not isinstance(instance_dir, (str, Path)):
        raise TypeError("instance_dir must be a string or Path")

    xy = load_matrices(resolve_under_root(instance_dir))["node_locs"].numpy()
    out = gdf.copy()
    out["node"] = _bind(gdf, xy).astype("Int64")
    return out


def connectivity_gdf(
    mapped: gpd.GeoDataFrame,
    instance_dir: str | Path,
    routes: torch.Tensor | list,
    agg: str | Callable = "sum",
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Return polygons with aggregated travel times in minutes and their pairwise matrix."""
    if not isinstance(mapped, gpd.GeoDataFrame):
        raise TypeError("mapped must be a GeoDataFrame")
    if "node" not in mapped.columns:
        raise ValueError("mapped must contain a 'node' column; call map_polygons first")
    if not isinstance(instance_dir, (str, Path)):
        raise TypeError("instance_dir must be a string or Path")
    if not isinstance(routes, (torch.Tensor, list)):
        raise TypeError("routes must be a torch.Tensor or list")
    if isinstance(routes, list) and not routes:
        raise ValueError("routes must not be empty")
    if isinstance(routes, torch.Tensor) and routes.ndim not in (2, 3):
        raise ValueError("routes must be a 2D or 3D tensor")
    if not isinstance(agg, str) and not callable(agg):
        raise TypeError("agg must be an aggregation name or callable")

    routes = as_route_tensor(routes).long()
    routes = routes if routes.ndim == 3 else routes[None]
    tensors = load_matrices(resolve_under_root(instance_dir))
    nodes = mapped["node"].dropna().astype(int)
    matrix = _matrix(routes, tensors)

    values = pd.DataFrame(matrix).agg(agg, axis=1).where(~np.isnan(matrix).all(1))
    out = mapped.copy()
    out["connectivity"] = nodes.map(values)
    full = pd.DataFrame(matrix[np.ix_(nodes.values, nodes.values)],
                        index=nodes.index, columns=nodes.index)

    to_polygon = dict(zip(nodes.values, nodes.index))
    out.attrs["routes"] = [[to_polygon[n] for n in r if n in to_polygon]
                           for r in routes[0].tolist()]
    return out, full
