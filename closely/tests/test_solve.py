import numpy as np
import pytest

import closest_pairs

@pytest.mark.parametrize("n_points,n_dim", [(2,2),(2,3),(3,2),(3,3)])
def test_solution(n_points, n_dim):
    X = np.random.random((1000,n_dim))
    pairs, distances = closest_pairs.solve(X,n=n_points)
    assert isinstance(pairs,np.ndarray)
    assert len(pairs) == n_points
    assert isinstance(distances, np.ndarray)
