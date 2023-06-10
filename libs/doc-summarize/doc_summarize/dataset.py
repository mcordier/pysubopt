"""Module for defining DocumentDataset class
"""
import os
from typing import Dict, List

import numpy as np
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


class DocumentDataset(BaseModel):

    """DocumentDataset which summarize all the data of a text
    document with different attributes.
    """

    X: list[str]
    X_train_counts: csr_matrix
    X_train_tf: csr_matrix
    tokenizer: PunktSentenceTokenizer
    bigram_vectorizer: CountVectorizer
    cosine_similarity_kernel: np.ndarray
    clusters: Dict[int, List[int]]

    class Config:
        arbitrary_types_allowed = True

    @property
    def V(self) -> List[int]:
        """Property to get the "index" of the set of
        sentences (isomorphism of the set to a numeral set)

        Returns:
            List[int]: Numeral set representing the index of the
            sentences
        """
        return list(range(self.X_train_counts.shape[0]))


def import_dataset(path: str) -> DocumentDataset:
    """Import a document dataset from a path.

    Args:
        path (str): Path of the document.

    Returns:
        DocumentDataset: DocumentDataset for the specified path
    """
    with open(path, "rt") as f:
        text = f.read()
    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train(text)
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    X = tokenizer.tokenize(text)
    bigram_vectorizer = CountVectorizer(
        ngram_range=(1, 2), token_pattern=r"\b\w+\b", min_df=1
    )
    X_train_counts = bigram_vectorizer.fit_transform(X)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    clusters = _get_cluster(X_train_tf)
    return DocumentDataset(
        X=X,
        X_train_counts=X_train_counts,
        X_train_tf=X_train_tf,
        tokenizer=tokenizer,
        bigram_vectorizer=bigram_vectorizer,
        cosine_similarity_kernel=cosine_similarity(X_train_tf, X_train_tf),
        clusters=clusters,
    )


def get_gold_summ_count(
    topics_file_name: str, dataset: DocumentDataset
) -> int:
    """Compute the gold sum count from a corpus of texts (topics)

    Args:
        topics_file_name (str): Description

    Returns:
        int: Gold sum count.
    """
    dir_name = topics_file_name.split(".")[0]
    path = "OpinosisDataset1.0_0/summaries-gold/{}".format(dir_name)
    K = len(os.listdir(path))
    gold_summ_txt = []
    gold_summ_count = []
    for i in range(1, K + 1):
        with open(path + "/{}.{}.gold".format(dir_name, i), "rt") as f:
            txt = f.read()
        gold_summ_txt.append(txt)
        a = dataset.tokenizer.tokenize(txt)
        if len(a) != 0:
            gold_summ_count.append(dataset.bigram_vectorizer.transform((a)))
    return gold_summ_count


def _get_cluster(X_train_tf, n_clusters=4):
    """Compute some clusters from a TF matrix representation of a text.

    Args:
        X_train_tf (TYPE): Description
        n_clusters (int, optional): Description

    Returns:
        TYPE: Description
    """
    # Clustering for div_R
    array = X_train_tf.toarray()
    # Initializing KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    # Fitting with inputs
    kmeans = kmeans.fit(array)
    # Predicting the clusters
    labels = kmeans.predict(array)
    cluster = {k: [] for k in range(n_clusters)}
    for i, label in enumerate(labels):
        cluster[label].append(i)
    return cluster
