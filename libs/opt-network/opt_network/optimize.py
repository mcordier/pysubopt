from functools import partial

import numpy as np
from opt_submodular import GreedySubmodularOptimizer, RandomSubset

from .obj_fun import expected_infected_nodes, covdiv_F
from .cost_fun import cost_select_nodes
from .dataset import GraphDataset


def optimize_covdiv_greedy(
    dataset: GraphDataset,
    budget: float,
    r: float = 1.,
    is_lazy: bool = True
):
    optimizer = GreedySubmodularOptimizer(
        fun=partial(covdiv_F, dataset=dataset),
        cost_fun=partial(cost_select_nodes, dataset=dataset),
        budget=budget,
        r=r,
        is_lazy=is_lazy
    )
    return optimizer.run(dataset.V)

def optimize_expected_greedy(
    dataset: GraphDataset,
    budget: float,
    model,
    T: int = 30,
    N: int = 20,
    r: float = 1.,
    is_lazy: bool = True
):
    optimizer = GreedySubmodularOptimizer(
        fun=partial(expected_infected_nodes, dataset=dataset, model=model, T=T, N=N),
        cost_fun=partial(cost_select_nodes, dataset=dataset),
        budget=budget,
        r=r,
        is_lazy=is_lazy
    )
    return optimizer.run(dataset.V)

def random_select(
    dataset: GraphDataset,
    budget: float,
):
    optimizer = RandomSubset(
        fun=lambda x: 0,
        cost_fun=partial(cost_select_nodes, dataset=dataset),
        budget=budget,
    )
    return optimizer.run(dataset.V)
