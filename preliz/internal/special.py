# pylint: disable=invalid-name
# pylint: disable=no-else-raise
import numba as nb
import numpy as np


@nb.vectorize(nopython=True, cache=True)
def erf(x):
    """
    Error function.

    Note:
    -----
    Adapted from Andreas Madsen's mathfn library
    """
    sign = 1
    if x < 0:
        sign = -1

    x = np.abs(x)

    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (
        ((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t
        + 0.254829592
    ) * t * np.exp(-x * x)

    return sign * y


ERFC_COF = np.array(
    [
        -2.8e-17,
        1.21e-16,
        -9.4e-17,
        -1.523e-15,
        7.106e-15,
        3.81e-16,
        -1.12708e-13,
        3.13092e-13,
        8.94487e-13,
        -6.886027e-12,
        2.394038e-12,
        9.6467911e-11,
        -2.27365122e-10,
        -9.91364156e-10,
        5.059343495e-9,
        6.529054439e-9,
        -8.5238095915e-8,
        1.5626441722e-8,
        1.303655835580e-6,
        -1.624290004647e-6,
        -2.0278578112534e-5,
        4.2523324806907e-5,
        3.66839497852761e-4,
        -9.46595344482036e-4,
        -9.561514786808631e-3,
        1.9476473204185836e-2,
        6.4196979235649026e-1,
        -1.3026537197817094,
    ]
)
ERFC_COF_LAST = ERFC_COF[-1]


@nb.vectorize(nopython=True, cache=True)
def erfccheb(y):
    d = 0.0
    dd = 0.0
    temp = 0.0
    t = 2.0 / (2.0 + y)
    ty = 4.0 * t - 2.0

    for i in range(len(ERFC_COF) - 1):
        temp = d
        d = ty * d - dd + ERFC_COF[i]
        dd = temp

    return t * np.exp(-y * y + 0.5 * (ERFC_COF_LAST + ty * d) - dd)


@nb.vectorize(nopython=True, cache=True)
def erfc(x):
    return erfccheb(x) if x >= 0.0 else 2.0 - erfccheb(-x)


@nb.vectorize(nopython=True, cache=True)
def erfcinv(p):
    """
    Invert complementary error function.

    Note:
    -----
    Adapted from Andreas Madsen's mathfn library
    """
    if p < 0.0 or p > 2.0:
        raise ValueError("Argument must be between 0 and 2")
    elif p == 0.0:
        return np.inf
    elif p == 2.0:
        return -np.inf
    else:
        pp = p if p < 1.0 else 2.0 - p
        t = np.sqrt(-2.0 * np.log(pp / 2.0))
        x = -0.70711 * ((2.30753 + t * 0.27061) / (1.0 + t * (0.99229 + t * 0.04481)) - t)

        err1 = erfc(x) - pp
        x += err1 / (1.12837916709551257 * np.exp(-x * x) - x * err1)
        err2 = erfc(x) - pp
        x += err2 / (1.12837916709551257 * np.exp(-x * x) - x * err2)

        return x if p < 1.0 else -x


@nb.njit(cache=True)
def erfinv(p):
    return -erfcinv(p + 1)


@nb.njit(cache=True)
def erfcx(x):
    return np.exp(x**2) * erfc(x)


@nb.vectorize(nopython=True, cache=True)
def beta(a, b):
    if a < 0 or b < 0:
        raise ValueError("Arguments must be positive.")
    elif a == 0 and b == 0:
        return np.inf
    elif a == 0 or b == 0:
        return np.inf

    return np.exp(betaln(a, b))


@nb.vectorize(nopython=True, cache=True)
def betaln(a, b):
    if a < 0 or b < 0:
        raise ValueError("Arguments must be positive.")
    elif a == 0 and b == 0:
        return np.inf
    elif a == 0 or b == 0:
        return np.inf

    return gammaln(a) + gammaln(b) - gammaln(a + b)


@nb.vectorize(nopython=True, cache=True)
def betacf(x, a, b):
    """
    Evaluates the continued fraction for incomplete beta function by modified Lentz's method.

    Note:
    -----
    Adapted from Andreas Madsen's mathfn library
    """

    fpmin = 1e-30
    c = 1
    qab = a + b
    qap = a + 1
    qam = a - 1
    d = 1 - qab * x / qap

    if abs(d) < fpmin:
        d = fpmin

    d = 1 / d
    h = d

    for m in range(1, 101):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d

        if abs(d) < fpmin:
            d = fpmin

        c = 1 + aa / c

        if abs(c) < fpmin:
            c = fpmin

        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d

        if abs(d) < fpmin:
            d = fpmin

        c = 1 + aa / c

        if abs(c) < fpmin:
            c = fpmin

        d = 1 / d
        del_ = d * c
        h *= del_

        if abs(del_ - 1.0) < 3e-7:
            break

    return h


@nb.vectorize(nopython=True, cache=True)
def betainc(a, b, x):
    """
    Returns the regularized incomplete beta function

    Note:
    -----
    Adapted from Andreas Madsen's mathfn library
    """
    if x < 0 or x > 1:
        raise ValueError("Third argument must be between 0 and 1.")
    elif a == 1 and b == 1:
        return x
    elif a == 0 or b == 0:
        return np.nan
    elif x == 0:
        return 0
    elif x == 1:
        return 1
    else:
        bt = np.exp(gammaln(a + b) - gammaln(a) - gammaln(b) + a * np.log(x) + b * np.log1p(-x))

        if x < (a + 1) / (a + b + 2):
            return bt * betacf(x, a, b) / a
        else:
            return 1 - bt * betacf(1 - x, b, a) / b


@nb.vectorize(nopython=True, cache=True)
def betainc_un(a, b, x):
    """
    Returns the "unregularized" incomplete beta function

    Note:
    -----
    Adapted from Andreas Madsen's mathfn library
    """
    return betainc(a, b, x) * beta(a, b)


@nb.vectorize(nopython=True, cache=True)
def betaincinv(a, b, p):
    """
    Returns the inverse of incomplete beta function

    Note:
    -----
    Adapted from Andreas Madsen's mathfn library
    """
    if p < 0 or p > 1:
        raise ValueError("Third argument must be between 0 and 1.")
    elif a == 1 and b == 1:
        return p
    elif p == 1:
        return 1
    elif p == 0:
        return np.nan
    else:
        EPS = 1e-8
        a1 = a - 1
        b1 = b - 1
        j = 0

        if a >= 1 and b >= 1:
            if p < 0.5:
                pp = p
            else:
                pp = 1 - p

            t = np.sqrt(-2 * np.log(pp))
            x = (2.30753 + t * 0.27061) / (1 + t * (0.99229 + t * 0.04481)) - t
            if p < 0.5:
                x = -x

            al = (x**2 - 3) / 6
            h = 2 / (1 / (2 * a - 1) + 1 / (2 * b - 1))
            w = (x * np.sqrt(al + h) / h) - ((1 / (2 * b - 1)) - (1 / (2 * a - 1))) * (
                al + 5 / 6 - 2 / (3 * h)
            )
            x = a / (a + b * np.exp(2 * w))
        else:
            lna = np.log(a / (a + b))
            lnb = np.log(b / (a + b))
            t = np.exp(a * lna) / a
            u = np.exp(b * lnb) / b
            w = t + u

            if p < t / w:
                x = (a * w * p) ** (1 / a)
            else:
                x = 1 - ((b * w * (1 - p)) ** (1 / b))

        afac = -betaln(a, b)

        for j in range(10):
            if x in (0, 1):
                return x

            err = betainc(a, b, x) - p
            t = np.exp(a1 * np.log(x) + b1 * np.log1p(-x) + afac)
            u = err / t
            t = u / (1 - 0.5 * min(1, u * (a1 / x - b1 / (1 - x))))
            x -= t

            if x <= 0:
                x = 0.5 * (x + t)
            if x >= 1:
                x = 0.5 * (x + t + 1)

            if abs(t) < EPS * x and j > 0:
                break

        return x


@nb.njit(cache=True)
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

    alpha_p = 1 / (z * (1 + (1 - z) ** 2 * poly))
    beta_p = 1 / (gamma(1 + 1 / alpha_p) / (mean))
    return alpha_p, beta_p


@nb.vectorize(nopython=True, cache=True)
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


@nb.vectorize(nopython=True, cache=True)
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
def xlogy(x, y):
    if x == 0:
        return 0.0
    else:
        return x * np.log(y)


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
    else:
        return x * y


@nb.vectorize(nopython=True, cache=True)
def cdf_bounds(prob, x, lower, upper):
    if x < lower:
        return 0
    elif x >= upper:
        return 1
    else:
        return prob


@nb.vectorize(nopython=True, cache=True)
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


@nb.vectorize(nopython=True, cache=True)
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


@nb.njit(cache=True)
def mean_and_std(data):
    n = len(data)
    mean = np.sum(data) / n

    sum_sq_diff = 0
    for x in data:
        sum_sq_diff += (x - mean) ** 2

    std = (sum_sq_diff / n) ** 0.5

    return mean, std


@nb.njit(cache=True)
def mean_sample(sample):
    return np.mean(sample)
