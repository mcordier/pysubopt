from typing import Dict, List

from pydantic import BaseModel
import numpy as np
import networkx as nx
import ndlib.models.epidemics as ep
from ndlib.utils import multi_runs
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

class GraphDataset(BaseModel):
    graph: nx.Graph
    dist: csr_matrix
    sim: np.ndarray
    degree_cent: Dict[int, float]
    clusters: Dict[int, List[int]]

    class Config:
        arbitrary_types_allowed = True

    @property
    def V(self):
        return list(self.graph.nodes)

def import_dataset(path: str) -> GraphDataset:
    graph = nx.read_edgelist(path, create_using=nx.Graph())
    graph = nx.convert_node_labels_to_integers(graph)
    dists = nx.adjacency_matrix(graph)
    sim = cosine_similarity(dists,dists)
    degree_cent = nx.degree_centrality(graph)
    clusters = _get_cluster(graph)
    return GraphDataset(
        graph=graph,
        dist=dists,
        sim=sim,
        degree_cent=degree_cent,
        clusters=clusters,
    )

def build_contagion_model(dataset, parameters):
    model = ep.SIRModel(dataset.graph)
    config = mc.Configuration()
    config.add_model_parameter('beta', parameters['beta'])
    config.add_model_parameter('gamma', parameters['gamma'])
    # config.add_model_parameter("infected", [])
    model.set_initial_status(config)
    return model

def _get_cluster(graph, n_clusters=4):
    '''
    Compute a n clustering
    Args:
        n_cluster (int) : number of cluster

    Returns:
        cluster (list) : list of the cluster index of each element

    '''
    adj_mat = nx.to_numpy_array(graph)
    sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=100)
    sc.fit(adj_mat)
    cluster = {k:[] for k in range(n_clusters)}
    for i in range(len(sc.labels_)):
        cluster[sc.labels_[i]].append(i)
    return cluster
