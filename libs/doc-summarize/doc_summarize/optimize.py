"""Module defining optimization functions for document summarization.
"""
from functools import partial
from typing import List

import numpy as np
from opt_submodular import (
    DoubleGreedySubmodularOptimizer,
    GreedySubmodularOptimizer,
)

from .cost_fun import cost_nbr_words
from .dataset import DocumentDataset
from .obj_fun import covdiv_F, get_f_MMR


def optimize_mmr_double_greedy(
    dataset: DocumentDataset, budget: float
) -> List[str]:
    """Document summarization function optimizing the MMR function with double
    greedy algorithm.

    Args:
        dataset (DocumentDataset): Document dataset.
        budget (float): Budget for the optimization problem.

    Returns:
        List[str]: Summary of the dataset.
    """
    optimizer = DoubleGreedySubmodularOptimizer(
        fun=partial(get_f_MMR, dataset=dataset),
        cost_fun=partial(cost_nbr_words, dataset=dataset),
        budget=budget,
    )
    return np.array(dataset.X)[optimizer.run(dataset.V).opt_subset]


def optimize_mmr_greedy(
    dataset: DocumentDataset,
    budget: float,
    r: float = 1.0,
    is_lazy: bool = True,
):
    """Document summarization function optimizing the MMR function
    with greedy algorithm.

    Args:
        dataset (DocumentDataset): Document dataset.
        budget (float): Budget for the optimization problem.
        r (float, optional): Scaling factor for the greedy algorithm.
        is_lazy (bool, optional): If true, will do the lazy algorithm version.
    """
    optimizer = GreedySubmodularOptimizer(
        fun=partial(get_f_MMR, dataset=dataset),
        cost_fun=partial(cost_nbr_words, dataset=dataset),
        budget=budget,
        r=r,
        is_lazy=is_lazy,
    )
    return np.array(dataset.X)[optimizer.run(dataset.V).opt_subset]


def optimize_covdiv_greedy(
    dataset: DocumentDataset,
    budget: float,
    r: float = 1.0,
    is_lazy: bool = True,
):
    """Document summarization function optimizing the covdiv function with greedy
    algorithm.

    Args:
        dataset (DocumentDataset): Document dataset.
        budget (float): Budget for the optimization problem.
        r (float, optional): Scaling factor for the greedy algorithm.
        is_lazy (bool, optional): If true, will do the lazy algorithm version.
    """
    optimizer = GreedySubmodularOptimizer(
        fun=partial(covdiv_F, dataset=dataset),
        cost_fun=partial(cost_nbr_words, dataset=dataset),
        budget=budget,
        r=r,
        is_lazy=is_lazy,
    )
    return np.array(dataset.X)[optimizer.run(dataset.V).opt_subset]
