from functools import partial

import numpy as np
from opt_submodular import GreedySubmodularOptimizer, DoubleGreedySubmodularOptimizer

from .obj_fun import get_f_MMR, covdiv_F
from .cost_fun import cost_nbr_words
from .dataset import DocumentDataset


def optimize_mmr_double_greedy(dataset: DocumentDataset, budget: float):
    optimizer = DoubleGreedySubmodularOptimizer(
        fun=partial(get_f_MMR, dataset=dataset),
        cost_fun=partial(cost_nbr_words, dataset=dataset),
        budget=budget
    )
    return np.array(dataset.X)[optimizer.run(dataset.V).opt_subset]

def optimize_mmr_greedy(
    dataset: DocumentDataset,
    budget: float,
    r: float = 1.,
    is_lazy: bool = True
):
    optimizer = GreedySubmodularOptimizer(
        fun=partial(get_f_MMR, dataset=dataset),
        cost_fun=partial(cost_nbr_words, dataset=dataset),
        budget=budget,
        r=r,
        is_lazy=is_lazy
    )
    return np.array(dataset.X)[optimizer.run(dataset.V).opt_subset]

def optimize_covdiv_greedy(
    dataset: DocumentDataset,
    budget: float,
    r: float = 1.,
    is_lazy: bool = True
):
    optimizer = GreedySubmodularOptimizer(
        fun=partial(covdiv_F, dataset=dataset),
        cost_fun=partial(cost_nbr_words, dataset=dataset),
        budget=budget,
        r=r,
        is_lazy=is_lazy
    )
    return np.array(dataset.X)[optimizer.run(dataset.V).opt_subset]
