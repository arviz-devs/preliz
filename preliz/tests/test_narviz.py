import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy.stats import gaussian_kde

from preliz.internal.narviz import hdi, kde, histogram, _dct1d, _bw_isj


def test_hdi():
    normal_sample = np.random.randn(50000)
    interval = hdi(normal_sample)
    assert_array_almost_equal(interval, [-1.8, 1.8], 1)


def test_kde():
    sample = np.random.randn(5000)
    grid, pdf = kde(sample)
    scipy_kde = gaussian_kde(sample)
    scipy_pdf = scipy_kde(grid)
    assert_array_almost_equal(pdf, scipy_pdf, decimal=1)


def test_dct1d():
    data = np.array([1, 2, 3, 4, 5])
    dct_result = _dct1d(data)

    expected_dct_result = np.array([17.0, -9.959, -1.236, -0.898, -3.236])

    assert_array_almost_equal(dct_result, expected_dct_result, decimal=3)


def test_bw_isj():
    sample = np.random.randn(5000)
    grid_counts, _, _ = histogram(sample, bins=256)
    x_range = sample.max() - sample.min()
    bandwidth = _bw_isj(sample, grid_counts=grid_counts, x_range=x_range)

    # Using a simple heuristic for expected bandwidth
    expected_bandwidth = 1.06 * np.std(sample) * len(sample) ** (-1 / 5)

    assert_array_almost_equal(bandwidth, expected_bandwidth, decimal=1)
