"""Summary."""
from typing import List

import numpy as np

from .dataset import DocumentDataset


def rouge_n(S: List[int], dataset: DocumentDataset, gold_summ_count) -> float:
    """Rouge-n measure for document summarize based on a "gold" standards".

    Args:
        S (List[int]): list of the index of the selected sentences S (subset)
        dataset (DocumentDataset): Dataset to summarize
        gold_sum_count (List[csr_matrix]): for each sparse matrix,
            gold_sum_count[k][i,j] is the number of times the bigram j occurs
            in the sentence in the summary k.

    Returns:
        float: Rouge-n measure of a subset

    """
    X_count_S_1 = dataset.X_train_counts[S].sum(
        axis=0
    )  # delete the sentence structure
    # X_count_S_1[s] : number of times the bigram s occurs in the summary S
    gold_summ_count_1 = [
        k.sum(axis=0) for k in gold_summ_count
    ]  # delete the sentence structure
    num = 0
    denom = 0
    for i in range(len(gold_summ_count_1)):
        for j in range(X_count_S_1.shape[1]):
            r_ei = gold_summ_count_1[i][0, j]
            if r_ei != 0:
                c_es = X_count_S_1[0, j]
                num += min(c_es, r_ei)
                denom += r_ei
    res = num / denom
    return res


# MMR
def get_f_MMR(S: List[int], dataset: DocumentDataset, lambda_MMR: float = 4.0):
    """Get the MMR value of a summary S.

    Args:
        S (List[int]: list of index of sentences
        dataset (DocumentDataset): Document dataset to summarize
        lambda_MMR (float, optional): Lambda MMR

    Returns:
        res: The MMR value of summary S

    """
    res = 0
    if len(S) == 0:
        return 0

    U = list(range(dataset.X_train_tf.shape[0]))
    S_c = [u for u in U if u not in S]

    for i in S_c:
        res += dataset.cosine_similarity_kernel[i, S].sum()

    if len(S) == 1:
        return res

    for i in S:
        S_without_i = S[:]
        S_without_i.remove(i)
        res -= (
            lambda_MMR
            * dataset.cosine_similarity_kernel[i, S_without_i].sum()
            / 2
        )

    return res


def covdiv_F(
    S: List[int], dataset: DocumentDataset, lambda_covdiv=4.0, alpha=0.6
):
    """Coverage-diversity function for documment summarization.

    Args:
        S (List[int]): List of index of sentences
        dataset (DocumentDataset): DocumentDataset
        lambda_covdiv (float, optional): Weight of diversity in the function.
        alpha (float, optional): Alpha

    Returns:
        float: Coverage-Diversity metric

    """

    def _cov_L(S):
        """Coverage function."""
        res = 0
        for x in dataset.V:
            res1 = dataset.cosine_similarity_kernel[x, S].sum()
            res2 = alpha * dataset.cosine_similarity_kernel[x, dataset.V].sum()
            res += min(res1, res2)
        return res

    def _div_R(S: List[int]):
        """Diversity function."""
        K = len(dataset.clusters)
        res = 0
        for k in range(K):
            S_inter_Pk = list(set(S) & set(dataset.clusters[k]))
            res1 = 0
            for j in S_inter_Pk:
                res1 += dataset.cosine_similarity_kernel[j, dataset.V].sum()
            res += np.sqrt(res1)
        return res

    if not len(S):
        return 0
    else:
        return _cov_L(S) + lambda_covdiv * _div_R(S)
