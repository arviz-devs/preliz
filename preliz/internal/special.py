# pylint: disable=invalid-name
import numba as nb
import numpy as np


@nb.njit
def betaln(a, b):
    return gammaln(a) + gammaln(b) - gammaln(a + b)


@nb.njit
def betafunc(a, b):
    return np.exp(betaln(a, b))


@nb.njit
def garcia_approximation(mean, sigma):
    """
    Approximate method of moments for Weibull distribution, provides good results for values of
    alpha larger than 0.83.

    Oscar Garcia. Simplified method-of-moments estimation for the Weibull distribution. 1981.
    New Zealand Journal of Forestry Science 11:304-306
    https://www.scionresearch.com/__data/assets/pdf_file/0010/36874/NZJFS1131981GARCIA304-306.pdf
    """
    ks = [-0.221016417, 0.010060668, 0.117358987, -0.050999126]  # pylint: disable=invalid-name
    z = sigma / mean  # pylint: disable=invalid-name

    poly = 0
    for idx, k in enumerate(ks):
        poly += k * z**idx

    alpha = 1 / (z * (1 + (1 - z) ** 2 * poly))
    beta = 1 / (gamma(1 + 1 / alpha) / (mean))
    return alpha, beta


@nb.vectorize(nopython=True)
def half_erf(x):
    """
    Error function for values of x >= 0, return 0 otherwise
    Equations 7.1.27 from Abramowitz and Stegun
    Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables
    """
    if x <= 0:
        return 0

    t = 1.0 / (
        1.0
        + 0.0705230784 * x
        + 0.0422820123 * x**2
        + 0.0092705272 * x**3
        + 0.0001520143 * x**4
        + 0.0002765672 * x**5
        + 0.0000430638 * x**6
    )
    approx = 1 - t**16

    return approx


@nb.vectorize(nopython=True)
def digamma(x):
    "Faster digamma function assumes x > 0."
    r = 0
    while x <= 5:
        r -= 1 / x
        x += 1
    f = 1 / (x * x)
    t = f * (
        -1 / 12.0
        + f
        * (
            1 / 120.0
            + f
            * (
                -1 / 252.0
                + f
                * (
                    1 / 240.0
                    + f * (-1 / 132.0 + f * (691 / 32760.0 + f * (-1 / 12.0 + f * 3617 / 8160.0)))
                )
            )
        )
    )

    return r + np.log(x) - 0.5 / x + t


@nb.njit
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


@nb.njit
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


@nb.vectorize(nopython=True)
def xlogy(x, y):
    if x == 0:
        return 0.0
    else:
        return x * np.log(y)


@nb.vectorize(nopython=True)
def cdf_bounds(prob, x, lower, upper):
    if x < lower:
        return 0
    elif x >= upper:
        return 1
    else:
        return prob


@nb.vectorize(nopython=True)
def ppf_bounds_disc(x_val, q, lower, upper):
    if q < 0:
        return np.nan
    elif q > 1:
        return np.nan
    elif q == 0:
        return lower - 1
    elif q == 1:
        return upper
    else:
        return x_val


@nb.vectorize(nopython=True)
def ppf_bounds_cont(x_val, q, lower, upper):
    if q < 0:
        return np.nan
    elif q > 1:
        return np.nan
    elif q == 0:
        return lower
    elif q == 1:
        return upper
    else:
        return x_val


@nb.njit
def mean_and_std(data):
    n = len(data)
    mean = np.sum(data) / n

    sum_sq_diff = 0
    for x in data:
        sum_sq_diff += (x - mean) ** 2

    std = (sum_sq_diff / n) ** 0.5

    return mean, std


@nb.njit
def mean_sample(sample):
    return np.mean(sample)
