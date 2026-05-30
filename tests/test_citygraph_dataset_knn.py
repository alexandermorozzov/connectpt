import torch
from torch_geometric.data import Data

from connectpt.routes_generator import citygraph_dataset as cgd


def test_torch_knn_graph_respects_flow_direction():
    pos = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
    ])

    source_to_target = cgd._torch_knn_graph(
        pos, 1, flow='source_to_target', directed=True)
    target_to_source = cgd._torch_knn_graph(
        pos, 1, flow='target_to_source', directed=True)

    assert {tuple(edge) for edge in source_to_target.t().tolist()} == {
        (1, 0),
        (0, 1),
        (1, 2),
    }
    assert {tuple(edge) for edge in target_to_source.t().tolist()} == {
        (0, 1),
        (1, 0),
        (2, 1),
    }


def test_apply_knn_graph_falls_back_when_torch_cluster_is_missing(monkeypatch):
    class MissingClusterKNN:
        def __call__(self, graph):
            raise ImportError("'knn_graph' requires 'torch-cluster'")

    pos = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
        [3.0, 0.0],
    ])
    monkeypatch.setattr(cgd, "_USE_TORCH_KNN_FALLBACK", False)

    graph = cgd._apply_knn_graph(
        Data(pos=pos),
        MissingClusterKNN(),
        1,
        flow='source_to_target',
        directed=False,
    )

    assert {tuple(edge) for edge in graph.edge_index.t().tolist()} == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
    }
