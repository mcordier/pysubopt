from typing import Dict, List

from pydantic import BaseModel

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from scipy.sparse import csr_matrix


class DocumentDataset(BaseModel):
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
    def V(self):
        return list(range(self.X_train_counts.shape[0]))

def import_dataset(path: str) -> DocumentDataset:
    text = open(path, encoding="utf8", errors='ignore')
    text = text.read()
    trainer = PunktTrainer()
    trainer.INCLUDE_ALL_COLLOCS = True
    trainer.train(text)
    tokenizer = PunktSentenceTokenizer(trainer.get_params())
    X = tokenizer.tokenize(text)
    bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                        token_pattern=r'\b\w+\b', min_df=1)
    X_train_counts = bigram_vectorizer.fit_transform(X)

    tf_transformer = TfidfTransformer(use_idf=True).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)

    clusters = _get_cluster(X_train_tf)
    keys = list(clusters.values())
    return DocumentDataset(
        X=X,
        X_train_counts=X_train_counts, 
        X_train_tf=X_train_tf,
        tokenizer=tokenizer,
        bigram_vectorizer=bigram_vectorizer,
        cosine_similarity_kernel=cosine_similarity(X_train_tf, X_train_tf),
        clusters=clusters
    )

def get_gold_summ_count(topics_file_name):
    dir_name = topics_file_name.split(".")[0]
    path = 'OpinosisDataset1.0_0/summaries-gold/{}'.format(dir_name)
    K = len([name for name in os.listdir(path)])
    # print(K)
    gold_summ_txt = []
    gold_summ_count = []
    for i in range(1, K + 1):
        text2 = open(path + '/{}.{}.gold'.format(dir_name, i), 'r')
        txt = text2.read()
        gold_summ_txt.append(txt)
        a = self.tokenizer.tokenize(txt)
        if len(a) != 0:
            gold_summ_count.append(self.bigram_vectorizer.transform((a)))
    return gold_summ_count

def _get_cluster(X_train_tf, n_clusters=4):
    # Clustering for div_R
    X = X_train_tf.toarray()
    # Initializing KMeans
    kmeans = KMeans(n_clusters=n_clusters)
    # Fitting with inputs
    kmeans = kmeans.fit(X)
    # Predicting the clusters
    labels = kmeans.predict(X)
    # Getting the cluster centers
    C = kmeans.cluster_centers_
    cluster = {k: [] for k in range(n_clusters)}
    for i in range(len(labels)):
        cluster[labels[i]].append(i)
    return cluster
