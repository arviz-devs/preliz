# pylint: disable=invalid-name
import numba as nb
import numpy as np


@nb.njit
def betaln(a, b):
    return gammaln(a) + gammaln(b) - gammaln(a + b)


@nb.njit
def betafunc(a, b):
    return np.exp(betaln(a, b))


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
def gammaln(z):
    log_z = np.log(z)
    return (
        z * log_z
        - z
        - 0.5 * log_z
        + 0.5 * np.log(2 * np.pi)
        + 1 / (12 * z)
        - 1 / (360 * z**3)
        + 1 / (1260 * z**5)
    )


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
    elif x > upper:
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
