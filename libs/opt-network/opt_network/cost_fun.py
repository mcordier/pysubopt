def cost_select_nodes(S, dataset, c=1):
    """
    Linear Cluster cost : compute the cost of selecting the nodes S
    The cost is linear with a neighbors/clustering measure

    Args:
      S (list): a list of nodes index
      c (float): cost for each neigbor

    Returns : cost of choosing nodes in S (float)
    """
    cost = 0
    for i in S:
        cost += len(dataset.graph[i]) * c
    return max(cost, 0.01)
