import numpy as np
import networkx as nx

def covdiv_F(S, dataset, lambda1=0.8, alpha=0.6):

    def div_R(S):
        K = len(dataset.clusters)
        res = 0
        for k in range(K):
            S_inter_Pk = list(set(S) & set(dataset.clusters[k]))
            res1 = 0
            for j in S_inter_Pk:
                res1 += 1 / len(list(dataset.V)) * dataset.sim[j,:].sum()
            res += np.sqrt(res1)
        return(res)

    def cov_L(S):
        res = 0
        for x in S:
            res += 1 / len(list(dataset.V)) * (dataset.sim[x,:]).sum()
        res = min(res, alpha * dataset.sim[:,:].sum())
        return res

    if not len(S):
        return 0
    else:
        return (cov_L(S) + lambda1 * div_R(S))


def expected_infected_nodes(S, dataset, model, T=30, N=20):
    '''
    Simulation for the expected number of total infected nodes
    during T with the inital set S

    Args:
        S (list): Initial infected nodes [list] (variable)
        T (int): number of iteration  (parameter)
        N (int): Number of simulation for Monte Carlo (parameter)

    Returns : expected number of total infected nodes (float) '''

    if not len(S):
        return(0)

    res = 0
    for i in range(N):

        #reinitialize model status and configuration
        model.reset()
        iterations = model.iteration_bunch(T)
        res += iterations[T-1]['node_count'][1]
        res += iterations[T-1]['node_count'][2]

    res /= N
    return(res)