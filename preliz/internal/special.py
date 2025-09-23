import numba as nb
import numpy as np


@nb.njit(cache=True)
def garcia_approximation(mean, sigma):
    """
    Approximate method of moments for Weibull distribution.

    The approximation is good for values of alpha larger than 0.83.

    Oscar Garcia. Simplified method-of-moments estimation for the Weibull distribution. 1981.
    New Zealand Journal of Forestry Science 11:304-306
    https://www.scionresearch.com/__data/assets/pdf_file/0010/36874/NZJFS1131981GARCIA304-306.pdf
    """
    ks = [-0.221016417, 0.010060668, 0.117358987, -0.050999126]
    z = sigma / mean

    poly = 0
    for idx, k in enumerate(ks):
        poly += k * z**idx

    alpha_p = 1 / (z * (1 + (1 - z) ** 2 * poly))
    beta_p = 1 / (gamma(1 + 1 / alpha_p) / (mean))
    return alpha_p, beta_p


@nb.njit(cache=True)
def gamma(z):
    p = [
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]

    z = z - 1
    x = np.ones_like(z) * 0.99999999999980993
    for i in range(8):
        x = x + p[i] / (z + i + 1)

    t = z + 7.5
    return np.sqrt(2 * np.pi) * t ** (z + 0.5) * np.exp(-t) * x


@nb.njit(cache=True)
def gammaln(x):
    cof = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ]
    stp = 2.5066282746310005
    fpf = 5.5
    x = x - 1.0
    tmp = x + fpf
    tmp = (x + 0.5) * np.log(tmp) - tmp
    ser = np.ones_like(x) * 1.000000000190015
    for j in range(6):
        x = x + 1
        ser = ser + cof[j] / x
    return tmp + np.log(stp * ser)


@nb.vectorize(nopython=True, cache=True)
def logit(x):
    if x == 0:
        return -np.inf
    elif x == 1:
        return np.inf
    if x < 0 or x > 1:
        return np.nan
    else:
        return np.log(x / (1 - x))


@nb.vectorize(nopython=True, cache=True)
def expit(x):
    if x >= 0:
        return 1 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))


@nb.vectorize(nopython=True, cache=True)
def xlogx(x):
    if x == 0:
        return 0.0
    else:
        return x * np.log(x)


@nb.vectorize(nopython=True, cache=True)
def xprody(x, y):
    if np.isinf(x):
        return 0
    if np.isinf(y):
        return 0
    else:
        return x * y


@nb.njit(cache=True)
def mean_and_std(data):
    n = len(data)
    mean = np.sum(data) / n

    sum_sq_diff = 0
    for x in data:
        sum_sq_diff += (x - mean) ** 2

    std = (sum_sq_diff / n) ** 0.5

    return mean, std
