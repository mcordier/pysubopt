"""Module for every defined cost functions.
"""
from typing import List

from .dataset import DocumentDataset


def cost_nbr_words(S: List[str], dataset: DocumentDataset) -> float:
    """
    Get the cost of Summary which is the number of words

    Args:
        S (List[str]: a list of senetences in summary
        dataset (DocumentDataset): Description

    Deleted Parameters:
        V_counts: sparse matrix to represent all sentences in grams counts

    Returns:
        float: cost of the subset
    """
    return dataset.X_train_counts[S].sum()
