"""Functions originally imported from ArviZ"""
import warnings

import numpy as np
from numba import njit

from scipy.fftpack import fft
from scipy.signal import convolve
from scipy.signal.windows import gaussian

from .optimization import _root
from .rcparams import rcParams


def hdi(ary, hdi_prob=None, skipna=True):
    """
    Calculate highest density interval (HDI) of array for given probability.

    The HDI is the minimum width Bayesian credible interval (BCI).

    Parameters
    ----------
    ary: array_like
        An array containing the values for which the HDI is to be computed.
    hdi_prob: float, optional
        Prob for which the highest density interval will be computed. Defaults to None,
        which results in the value of rcParams["stats.ci_prob"] being used.

    Returns
    -------
    np.ndarray with lower and upper values of the interval.
    """
    if hdi_prob is None:
        hdi_prob = rcParams["stats.ci_prob"]

    if not 1 >= hdi_prob > 0:
        raise ValueError("The value of hdi_prob should be in the interval (0, 1]")

    ary = np.ravel(ary)
    if skipna:
        nans = np.isnan(ary)
        if not nans.all():
            ary = ary[~nans]
    n = len(ary)

    ary = np.sort(ary)
    interval_idx_inc = int(np.floor(hdi_prob * n))
    n_intervals = n - interval_idx_inc
    interval_width = np.subtract(ary[interval_idx_inc:], ary[:n_intervals], dtype=np.float64)

    if len(interval_width) == 0:
        raise ValueError("Too few elements for interval calculation. ")

    min_idx = np.argmin(interval_width)
    hdi_min = ary[min_idx]
    hdi_max = ary[min_idx + interval_idx_inc]
    hdi_interval = np.array([hdi_min, hdi_max])

    return hdi_interval


def kde(x):
    x = x[np.isfinite(x)]
    if x.size == 0 or np.all(x == x[0]):
        warnings.warn("Your data appears to have a single value or no finite values")

        return np.zeros(2), np.array([np.nan] * 2)

    grid_len = 256
    # Preliminary calculations
    x_min = x.min()
    x_max = x.max()
    x_range = x_max - x_min

    # Determine grid
    grid_min = x_min
    grid_max = x_max

    grid_counts, _, grid_edges = histogram(x, grid_len, (grid_min, grid_max))

    # Bandwidth estimation

    band_w = _bw_isj(x, grid_counts=grid_counts, x_range=x_range)

    # Density estimation
    grid, pdf = _kde_convolution(x, band_w, grid_edges, grid_counts, grid_len)

    return grid, pdf


@njit(cache=True)
def histogram(data, bins, range_hist=None):
    """Jitted histogram.

    Parameters
    ----------
    data : array-like
        Input data. Passed as first positional argument to ``np.histogram``.
    bins : int or array-like
        Passed as keyword argument ``bins`` to ``np.histogram``.
    range_hist : (float, float), optional
        Passed as keyword argument ``range`` to ``np.histogram``.

    Returns
    -------
    hist : array
        The number of counts per bin.
    density : array
        The density corresponding to each bin.
    bin_edges : array
        The edges of the bins used.
    """
    hist, bin_edges = np.histogram(data, bins=bins, range=range_hist)
    hist_dens = hist / (hist.sum() * np.diff(bin_edges))
    return hist, hist_dens, bin_edges


def _kde_convolution(x, band_w, grid_edges, grid_counts, grid_len):
    """Kernel density with convolution.

    One dimensional Gaussian kernel density estimation via convolution of the binned relative
    frequencies and a Gaussian filter. This is an internal function used by `kde()`.
    """
    # Calculate relative frequencies per bin
    bin_width = grid_edges[1] - grid_edges[0]
    freq = grid_counts / bin_width / len(x)

    # Bandwidth must consider the bin width
    band_w /= bin_width

    grid = (grid_edges[1:] + grid_edges[:-1]) / 2

    kernel_n = int(band_w * 2 * np.pi)
    if kernel_n == 0:
        kernel_n = 1

    kernel = gaussian(kernel_n, band_w)

    npad = int(grid_len / 5)
    freq = np.concatenate([freq[npad - 1 :: -1], freq, freq[grid_len : grid_len - npad - 1 : -1]])
    pdf = convolve(freq, kernel, mode="same", method="direct")[npad : npad + grid_len]
    pdf /= band_w * (2 * np.pi) ** 0.5

    return grid, pdf


def _bw_isj(x, grid_counts=None, x_range=None):
    """Improved Sheather-Jones bandwidth estimation.

    Improved Sheather and Jones method as explained in [1]_. This method is used internally by the
    KDE estimator, resulting in saved computation time as minimums, maximums and the grid are
    pre-computed.

    References
    ----------
    .. [1] Kernel density estimation via diffusion.
       Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
       Ann. Statist. 38 (2010), no. 5, 2916--2957.
    """
    x_len = len(x)
    grid_len = len(grid_counts) - 1
    # Discrete cosine transform of the data
    a_k = _dct1d(grid_counts / x_len)

    k_sq = np.arange(1, grid_len) ** 2
    a_sq = a_k[range(1, grid_len)] ** 2
    return _root(x_len, k_sq, a_sq, x) ** 0.5 * x_range


def _dct1d(x):
    """Discrete Cosine Transform in 1 Dimension.

    Parameters
    ----------
    x : numpy array
        1 dimensional array of values for which the
        DCT is desired

    Returns
    -------
    output : DTC transformed values
    """
    x_len = len(x)

    even_increasing = np.arange(0, x_len, 2)
    odd_decreasing = np.arange(x_len - 1, 0, -2)

    x = np.concatenate((x[even_increasing], x[odd_decreasing]))

    w_1k = np.r_[1, (2 * np.exp(-(0 + 1j) * (np.arange(1, x_len)) * np.pi / (2 * x_len)))]
    output = np.real(w_1k * fft(x))

    return output
