"""Module defining cost functions.
"""
from typing import List

from .dataset import GraphDataset


def cost_select_nodes(
    S: List[int], dataset: GraphDataset, c: float = 1.0
) -> float:
    """
    Compute the cost of selecting the nodes S (linear)

    Args:
        S (List[int]): Subset of nodes (list of index)
        dataset (GraphDataset): Graph dataset
        c (float, optional): Cost of choosing nodes in S (float)

    Returns:
        float: Cost of selecting the subset of nodes.
    """
    cost = 0
    for i in S:
        cost += len(dataset.graph[i]) * c
    return max(cost, 0.01)
