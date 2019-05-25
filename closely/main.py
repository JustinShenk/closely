import math

import numpy as np
from scipy.spatial.distance import pdist, squareform


def solution(array: np.ndarray, n=1, metric="euclidean", max_dist=None, quantile=0.0037):
    """Solve the closest pairs problem.
    Args:
        array (np.ndarray): N x M

    """
    pairs = []
    distances = []

    distance_matrix = pdist(array, metric=metric)
    if not max_dist:
        max_dist = np.quantile(distance_matrix, quantile)

    # Convert to square form
    distance_matrix = squareform(distance_matrix)

    for idx, img in enumerate(distance_matrix):
        for idx2, dist in enumerate(img):
            if idx != idx2 and dist <= (max_dist or np.inf):
                print(idx, idx2, dist)
                pair = sorted([idx, idx2])
                pairs.append(pair)
                distances.append(dist)

    pairs, indices = np.unique(pairs, axis=0, return_index=True)
    return pairs, np.array(distances)[indices]
