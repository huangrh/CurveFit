# -*- coding: utf-8 -*-
"""
    test utils
    ~~~~~~~~~~

    Test utils module
"""

import numpy as np
import pandas as pd
import pytest
import curvefit
import curvefit.utils as utils
from curvefit.legacy.utils import neighbor_mean_std as old_algorithm
from curvefit.utils import neighbor_mean_std as new_algorithm


def generate_testing_problem(locations=("USA", "Europe", "Asia"),
                             timelines=(10, 20, 30),
                             seed=42):
    """
    Generates sample problem for testing utils.neighbor_mean_std function. The columns are:
        - 'group': group parameter,
        - 'far_out': first axis,
        - 'num_data': second axis,
        - 'residual': value to aggregate, generated from U[0, 1]

    Args:
        locations: Set{String}
            Locations, group parameter.
        timelines: Set{int}
            How many data points to generate per location
        seed: int
            Random seed

    Returns:
        new_df: pd.DataFrame
            Random dataset suitable for testing neighbor_mean_std function.
    """

    far_out = []
    num_data = []
    location = []
    residual = []
    np.random.seed(seed)
    for t, place in zip(timelines, locations):
        for horizon in np.arange(1, t):
            far_out += [horizon] * (t - horizon)
            num_data += np.arange(1, t - horizon + 1).tolist()
            location += [place] * (t - horizon)
            residual += np.random.rand(t - horizon).tolist()
    new_df = pd.DataFrame({
        'group': location,
        'far_out': far_out,
        'num_data': num_data,
        'residual': residual,
    })
    return new_df


def test_neighbor_mean_std_consistent_with_old_algorithm():
    """
    Compares that new (Aleksei's) algorithm works consistently with old (Peng's) algorithm

    Returns:
        None
    """
    data = generate_testing_problem()
    old_alg_result = old_algorithm(data,
                                   col_axis=['far_out', 'num_data'],
                                   col_val='residual',
                                   col_group='group',
                                   radius=[2, 2]
                                   )
    new_alg_result = new_algorithm(data,
                                   col_axis=['far_out', 'num_data'],
                                   col_val='residual',
                                   col_group='group',
                                   radius=[2, 2]
                                   )

    assert np.allclose(old_alg_result["residual_mean"], new_alg_result["residual_mean"])
    assert np.allclose(old_alg_result["residual_std"], new_alg_result["residual_std"])
    return None


@pytest.mark.parametrize(('sizes', 'indices'),
                         [(np.array([1, 1, 1]), [np.array([0]),
                                                 np.array([1]),
                                                 np.array([2])]),
                          (np.array([1, 2, 3]), [np.array([0]),
                                                 np.array([1, 2]),
                                                 np.array([3, 4, 5])])])
def test_sizes_to_indices(sizes, indices):
    my_indices = utils.sizes_to_indices(sizes)
    print(my_indices)
    assert all([np.allclose(indices[i], my_indices[i])
                for i in range(sizes.size)])

@pytest.fixture
def data():
    return pd.DataFrame({
        't': np.arange(5),
        'group': 'All',
        'obs': np.ones(5),
        'cov': np.zeros(5)
    })

@pytest.mark.parametrize('func', [lambda x: 1 / (1 + x),
                                  lambda x: x**2])
def test_get_obs_se(data, func):
    result = utils.get_obs_se(data, 't', func=func)
    assert np.allclose(result['obs_se'], func(data['t']))


@pytest.mark.parametrize('t', [np.arange(5)])
@pytest.mark.parametrize(('start_day', 'end_day', 'pred_fun'),
                         [(1, 3, curvefit.derf)])
@pytest.mark.parametrize(('mat1', 'mat2', 'result'),
                         [(np.ones(5), np.ones(5), np.ones(5)),
                          (np.arange(5), np.ones(5),
                           np.array([1.0, 1.0, 1.5, 3.0, 4.0]))])
def test_convex_combination(t, mat1, mat2, pred_fun, start_day, end_day,
                            result):
    my_result = utils.convex_combination(t, mat1, mat2, pred_fun,
                                         start_day=start_day,
                                         end_day=end_day)

    assert np.allclose(result, my_result)

@pytest.mark.parametrize(('w1', 'w2', 'pred_fun'),
                         [(0.3, 0.7, curvefit.derf)])
@pytest.mark.parametrize(('mat1', 'mat2', 'result'),
                         [(np.ones(5), np.ones(5), np.ones(5)),
                          (np.ones(5), np.zeros(5), np.ones(5)*0.3),
                          (np.zeros(5), np.ones(5), np.ones(5)*0.7)])
def test_model_average(mat1, mat2, w1, w2, pred_fun, result):
    my_result = utils.model_average(mat1, mat2, w1, w2, pred_fun)
    assert np.allclose(result, my_result)


@pytest.mark.parametrize('mat', [np.arange(9).reshape(3, 3)])
@pytest.mark.parametrize(('radius', 'result'),
                         [((0, 0), np.arange(9).reshape(3, 3)),
                          ((1, 1), np.array([[ 8, 15, 12],
                                            [21, 36, 27],
                                            [20, 33, 24]]))])
def test_convolve_sum(mat, radius, result):
    my_result = utils.convolve_sum(mat, radius=radius)
    assert np.allclose(result, my_result)


def test_df_to_mat():
    df = pd.DataFrame({
        'val': np.ones(5),
        'axis0': np.arange(5, dtype=int),
        'axis1': np.arange(5, dtype=int)
    })

    my_result, indices, axis = utils.df_to_mat(df, 'val', ['axis0', 'axis1'],
                                               return_indices=True)
    assert np.allclose(my_result[indices[:, 0], indices[:, 1]], 1.0)
