"""Module defining the objective functions for the submodular optimization.
"""
from typing import List

import ndlib.models.epidemics as ep
import numpy as np

from .dataset import GraphDataset


def covdiv_F(
    S: List[int],
    dataset: GraphDataset,
    lambda_covdiv: float = 0.8,
    alpha: float = 0.6,
) -> float:
    """Coverage-diversity function for graph optimization.

    Args:
        S (List[int]): List of index of nodes
        dataset (GraphDataset): GraphDataset
        lambda_covdiv (float, optional): Weight of diversity in the function.
        alpha (float, optional): Alpha

    Returns:
        float: Coverage-Diversity metric
    """

    def div_R(S):
        """Diversity function."""
        K = len(dataset.clusters)
        res = 0
        for k in range(K):
            S_inter_Pk = list(set(S) & set(dataset.clusters[k]))
            res1 = 0
            for j in S_inter_Pk:
                res1 += 1 / len(list(dataset.V)) * dataset.sim[j, :].sum()
            res += np.sqrt(res1)
        return res

    def cov_L(S):
        """Coverage function."""
        res = 0
        for x in S:
            res += 1 / len(list(dataset.V)) * (dataset.sim[x, :]).sum()
        res = min(res, alpha * dataset.sim[:, :].sum())
        return res

    if len(S):
        return cov_L(S) + lambda_covdiv * div_R(S)

    return 0


def expected_infected_nodes(
    S: List[int],
    dataset: GraphDataset,
    model: ep.SIRModel,
    T: int = 30,
    N: int = 20,
) -> float:
    """
    Simulation for the expected number of total infected nodes
    during T with the inital set S

    Args:
        S (List[int]): List of index of nodes
        dataset (GraphDataset): GraphDataset
        model (ep.SIRModel): Contagion model
        T (int, optional): Number of timesteps in the contagion.
        N (int, optional): Number of iterations (Monte-Carlo)

    Returns:
        float: expected number of total infected nodes
    """

    if not len(S):
        return 0

    res = 0
    for _ in range(N):
        # reinitialize model status and configuration
        model.reset()
        iterations = model.iteration_bunch(T)
        res += iterations[T - 1]["node_count"][1]
        res += iterations[T - 1]["node_count"][2]

    res /= N
    return res
