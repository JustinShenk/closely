from typing import Optional

import numpy as np
from scipy.spatial.distance import cdist


def solution(array: np.ndarray, n:Optional[int]=None, metric="euclidean", quantile:Optional[float]=None):
    """Solve the closest pairs problem.
    Args:
        array (np.ndarray): N x M
        n (int, optional): number of closest pairs
        metric (str): distance metric
        quantile (float, optional): between 0 and 1
    Returns:
        pairs (list of tuples): closest pairs of points
        distances (np.ndarray): distances between pairs

    """
    if quantile is not None:
        index = get_index_of_quantile(array, quantile)
        pairs, distances = closest_k_pairs(array, kth=index, metric=metric)
    else:
        pairs, distances = closest_k_pairs(array, kth=n, metric=metric)
    return pairs, distances


def closest_k_pairs(array:np.ndarray, kth:int=3, metric:str='euclidean'):
    """Get closest k-pairs in a matrix.
    Args:
        array (np.ndarray): n instances x m features matrix
        kth (int): k lowest pairs
        metric (str): distance metric
    Returns:
        pairs (list of tuples): coordinates of nearest pairs, ordered
        distances (np.ndarray): 1-d array of distances for each pair, sorted

    """
    # Calculate distance matrix
    dist_mat = distance_matrix(array, metric=metric)

    # Replace diagonal and upper triangle with inf
    dist_mat[np.tril_indices(dist_mat.shape[0], -1)] = np.inf
    np.fill_diagonal(dist_mat, np.inf)

    coord1, coord2 = np.unravel_index(np.argpartition(dist_mat, kth=kth, axis=None), dist_mat.shape)

    pairs = list(zip(coord1[:kth], coord2[:kth]))
    distances = dist_mat[coord1[:kth], coord2[:kth]]
    return pairs, distances


def get_index_of_quantile(dist_mat:np.ndarray, quantile:float):
    """Returns index of `quantile` in `dist_mat`."""
    flat_dist_mat = dist_mat.flatten()
    flat_dist_mat.sort()

    value = np.quantile(flat_dist_mat, quantile)
    index = np.searchsorted(flat_dist_mat, value)
    return index


def seriation(Z, N, cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z

        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index - N, 0])
        right = int(Z[cur_index - N, 1])
        return (seriation(Z, N, left) + seriation(Z, N, right))


def compute_serial_matrix(dist_mat, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)

        compute_serial_matrix transforms a distance matrix into
        a sorted distance matrix according to the order implied
        by the hierarchical tree (dendrogram)
    '''
    try:
        from fastcluster import linkage
    except ImportError:
        raise("fastcluster is not installed. Install it with 'pip install fastcluster'")
    N = len(dist_mat)
    res_linkage = linkage(dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N - 2)
    seriated_dist = np.zeros((N, N))
    a, b = np.triu_indices(N, k=1)
    seriated_dist[a, b] = dist_mat[[res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b, a] = seriated_dist[a, b]
    return seriated_dist, res_order, res_linkage



def distance_matrix(array:np.ndarray, metric:str='euclidean', ordered:bool=False):
    dist_mat = cdist(array, array, metric=metric)

    if ordered:
        ordered_dist_mat, _, _ = order_matrix(dist_mat)
        return ordered_dist_mat
    return dist_mat


def order_matrix(dist_mat:np.ndarray, method:str="ward"):
    """Order the matrix by hierarchical clustering.
    Args:
        dist_mat
        method (str): Clustering method, one of 'ward', 'single', 'average', 'complete'
    Returns:
        ordered_dist_mat (np.ndarray): ordered distance matrix
        res_order (list): resulting order
        res_linkage (np.ndarray): resulting linkage
    """
    ordered_dist_mat, res_order, res_linkage = compute_serial_matrix(dist_mat, method)
    return ordered_dist_mat, res_order, res_linkage


def show_linkage(dist_mat:np.ndarray):
    """Plot and show linkage of distance matrix.
    Args:
        dist_mat (np.ndarray): ordered distance matrix
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise("Error: Install matplotlib with 'pip install matplotlib'")
    N = len(dist_mat)

    if np.inf in dist_mat:
        # Make symmetric
        ordered_dist_mat = np.minimum(dist_mat, dist_mat.transpose())

    plt.pcolormesh(dist_mat)
    plt.xlim([0,N])
    plt.ylim([0,N])
    plt.show()


