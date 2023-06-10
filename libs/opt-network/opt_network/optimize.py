"""Module defining optimization functions for graph optimization.
"""
from functools import partial

from opt_submodular import GreedySubmodularOptimizer, RandomSubset

from .cost_fun import cost_select_nodes
from .dataset import GraphDataset
from .obj_fun import covdiv_F, expected_infected_nodes


def optimize_covdiv_greedy(
    dataset: GraphDataset, budget: float, r: float = 1.0, is_lazy: bool = True
):
    """Optimizing the covdiv_ function with greedy algorithm on the graph.

    Args:
        dataset (GraphDataset): Graph dataset.
        budget (float): Budget for the optimization problem.
        r (float, optional): Scaling factor for the greedy algorithm.
        is_lazy (bool, optional): If true, will do the lazy algorithm version.
    """
    optimizer = GreedySubmodularOptimizer(
        fun=partial(covdiv_F, dataset=dataset),
        cost_fun=partial(cost_select_nodes, dataset=dataset),
        budget=budget,
        r=r,
        is_lazy=is_lazy,
    )
    return optimizer.run(dataset.V)


def optimize_expected_greedy(
    dataset: GraphDataset,
    budget: float,
    model,
    T: int = 30,
    N: int = 20,
    r: float = 1.0,
    is_lazy: bool = True,
):
    """Optimizing the expected number of infections
    with greedy algorithm on the graph.

    Args:
        dataset (GraphDataset): Graph dataset.
        budget (float): Budget for the optimization problem.
        model (ep.SIRModel): Contagion model
        T (int, optional): Number of timesteps in the contagion.
        N (int, optional): Number of iterations (Monte-Carlo)
        r (float, optional): Scaling factor for the greedy algorithm.
        is_lazy (bool, optional): If true, will do the lazy algorithm version.
    """
    optimizer = GreedySubmodularOptimizer(
        fun=partial(
            expected_infected_nodes, dataset=dataset, model=model, T=T, N=N
        ),
        cost_fun=partial(cost_select_nodes, dataset=dataset),
        budget=budget,
        r=r,
        is_lazy=is_lazy,
    )
    return optimizer.run(dataset.V)


def random_select(
    dataset: GraphDataset,
    budget: float,
):
    """Select a ranom subset from dataset with the right budget

    Args:
        dataset (GraphDataset): Graph dataset
        budget (float): Budget for the optimization problem.
    """
    optimizer = RandomSubset(
        fun=lambda x: 0,
        cost_fun=partial(cost_select_nodes, dataset=dataset),
        budget=budget,
    )
    return optimizer.run(dataset.V)
