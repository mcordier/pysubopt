"""Module defining the GraphDataset used in this lib.
"""
from typing import Dict, List

import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
import numpy as np
from pydantic import BaseModel
from scipy.sparse import csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity


class GraphDataset(BaseModel):

    """GraphDataset class with useful attributes for the optimization
    process.
    """

    graph: nx.Graph
    dist: csr_matrix
    sim: np.ndarray
    degree_cent: Dict[int, float]
    clusters: Dict[int, List[int]]

    class Config:
        arbitrary_types_allowed = True

    @property
    def V(self) -> List[int]:
        """Property to get the "index" of the set of
        nodes (isomorphism of the set to a numeral set)

        Returns:
            List[int]: Numeral set representing the index of the
            nodes
        """
        return list(self.graph.nodes)


def import_dataset(path: str) -> GraphDataset:
    """Import a graph dataset from a path.

    Args:
        path (str): Path of the graph.

    Returns:
        GraphDataset: GraphDataset for the specified path
    """
    graph = nx.read_edgelist(path, create_using=nx.Graph())
    graph = nx.convert_node_labels_to_integers(graph)
    dists = nx.adjacency_matrix(graph)
    sim = cosine_similarity(dists, dists)
    degree_cent = nx.degree_centrality(graph)
    clusters = _get_cluster(graph)
    return GraphDataset(
        graph=graph,
        dist=dists,
        sim=sim,
        degree_cent=degree_cent,
        clusters=clusters,
    )


def build_contagion_model(
    dataset: GraphDataset, parameters: Dict[str, float]
) -> ep.SIRModel:
    """Build a contagion model based a graph dataset, and some contagion parameters.

    Args:
        dataset (GraphDataset): Graph dataset.
        parameters (Dict[str, float]): Parameters for the contagion model.

    Returns:
        ep.SIRModel: Contagion model
    """
    model = ep.SIRModel(dataset.graph)
    config = mc.Configuration()
    config.add_model_parameter("beta", parameters["beta"])
    config.add_model_parameter("gamma", parameters["gamma"])
    # config.add_model_parameter("infected", [])
    model.set_initial_status(config)
    return model


def _get_cluster(graph, n_clusters=4):
    """
    Compute a n clustering on graph nodes based on the
    adjency matrix.

    Args:
        graph (nx.Graph): Description
        n_clusters (int, optional): Number of clusters

    Returns:
        cluster (list): list of the cluster index of each element
    """
    adj_mat = nx.to_numpy_array(graph)
    sc = SpectralClustering(n_clusters, affinity="precomputed", n_init=100)
    sc.fit(adj_mat)
    cluster = {k: [] for k in range(n_clusters)}
    for i in range(len(sc.labels_)):
        cluster[sc.labels_[i]].append(i)
    return cluster
