# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb
from scipy.special import comb

from .distributions import Continuous
from ..internal.distribution_helper import all_not_none, eps, from_precision, to_precision
from ..internal.optimization import optimize_ml, optimize_moments
from ..internal.special import beta, betainc, betaincinv, cdf_bounds, ppf_bounds_cont, gamma


class SkewStudentT(Continuous):
    r"""
    Jones and Faddy's Skewed Student-t distribution.

    The pdf of this distribution is

    .. math::

        f(t)=f(t ; a, b)=C_{a, b}^{-1}\left\{1+\frac{t}{\left(a+b+t^2\right)^{1 / 2}}
        \right\}^{a+1 / 2}\left\{1-\frac{t}{\left(a+b+t^2\right)^{1 / 2}}\right\}^{b+1 / 2}

    where

    .. math::

        C_{a, b}=2^{a+b-1} B(a, b)(a+b)^{1 / 2}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import SkewStudentT
        az.style.use('arviz-doc')
        mus = [2., 2., 4.]
        sigmas = [1., 2., 2.]
        a_s = [1., 1., 2.]
        b_s = [2., 3., 3.]
        for mu, sigma, a, b in zip(mus, sigmas, a_s, b_s):
            SkewStudentT(mu, sigma, a, b).plot_pdf(support=(-10,6))

    ========  ==========================================
    Support   :math:`x \in \mathbb{R}`
    Mean      .. math::
                  \mu + \sigma\frac{(a-b) \sqrt{(a+b)}}{2}\frac{\Gamma\left(a-\frac{1}{2}\right)
                  \Gamma\left(b-\frac{1}{2}\right)}{\Gamma(a) \Gamma(b)}
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location Parameter
    sigma : float
        Scale parameter (sigma > 0). Converges to the standard deviation as a and b approach close
    a : float
        First Shape Parameter (a > 0)
    b : float
        Second Shape Parameter (b > 0)
    lam : float
        Scale Parameter (lam > 0). Converges to the precision as a and b approach close

    Notes
    -----
    If a > b, the distribution is positively skewed, and if a < b, the distribution
    is negatively skewed. If a = b, the distribution is StudentT with
    nu (degrees of freedom) = 2a.

    """

    def __init__(self, mu=None, sigma=None, a=None, b=None, lam=None):
        super().__init__()
        self._parametrization(mu, sigma, a, b, lam)

    def _parametrization(self, mu=None, sigma=None, a=None, b=None, lam=None):
        if all_not_none(sigma, lam):
            raise ValueError("Incompatible parametrization. Either use sigma or lam.")
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        self.support = (-np.inf, np.inf)
        self.param_names = ("mu", "sigma", "a", "b")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf), (eps, np.inf), (eps, np.inf))
        if lam is not None:
            self.lam = lam
            sigma = from_precision(lam)
            self.param_names = ("mu", "lam", "a", "b")
        if all_not_none(mu, sigma, a, b):
            self._update(mu, sigma, a, b)

    def _update(self, mu, sigma, a, b):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.lam = to_precision(sigma)
        self.a = np.float64(a)
        self.b = np.float64(b)
        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma, self.a, self.b)
        elif self.param_names[1] == "lam":
            self.params = (self.mu, self.lam, self.a, self.b)
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.mu, self.sigma, self.a, self.b))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.mu, self.sigma, self.a, self.b, -np.inf, np.inf)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.mu, self.sigma, self.a, self.b, -np.inf, np.inf)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu, self.sigma, self.a, self.b)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.sigma, self.a, self.b)

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        return -np.trapz(np.exp(logpdf) * logpdf, x_values)

    def mean(self):
        return (
            (self.a - self.b) * (self.a + self.b) ** 0.5 * gamma(self.a - 0.5) * gamma(self.b - 0.5)
        ) / (2 * gamma(self.a) * gamma(self.b)) * self.sigma + self.mu

    def median(self):
        return self.ppf(0.5)

    def var(self):
        nu = (
            (self.a + self.b) ** 0.5
            / (2 * beta(self.a, self.b))
            * (
                comb(1, np.arange(2))
                * np.where(np.arange(1 + 1) % 2 > 0, -1, 1)
                * beta(self.a + 0.5 - np.arange(2), self.b - 0.5 + np.arange(2))
            ).sum()
        )
        return (
            np.where(
                np.isfinite(nu),
                (self.a + self.b)
                / (4 * beta(self.a, self.b))
                * (
                    comb(2, np.arange(3))
                    * np.where(np.arange(3) % 2 > 0, -1, 1)
                    * beta(self.a + 1 - np.arange(3), self.b - 1 + np.arange(3))
                ).sum()
                - nu**2,
                np.inf,
            )
            * self.sigma**2
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        nu1 = (
            (self.a + self.b) ** 0.5
            / (2 * beta(self.a, self.b))
            * (
                comb(1, np.arange(2))
                * np.where(np.arange(2) % 2 > 0, -1, 1)
                * beta(self.a + 0.5 - np.arange(2), self.b - 0.5 + np.arange(2))
            ).sum()
        )
        nu2 = ((self.a + self.b) / (4 * beta(self.a, self.b))) * (
            comb(2, np.arange(3))
            * np.where(np.arange(3) % 2 > 0, -1, 1)
            * beta(self.a + 1 - np.arange(3), self.b - 1 + np.arange(3))
        ).sum() - nu1**2
        nu3 = (
            ((self.a + self.b) ** 1.5 / (8 * beta(self.a, self.b)))
            * (
                comb(3, np.arange(4))
                * np.where(np.arange(4) % 2 > 0, -1, 1)
                * beta(self.a + 1.5 - np.arange(4), self.b - 1.5 + np.arange(4))
            ).sum()
            - nu1**3
            - 3 * nu1 * nu2
        )
        return nu3 / nu2**1.5

    def kurtosis(self):
        nu1 = ((self.a + self.b) ** 0.5 / (2 * beta(self.a, self.b))) * (
            comb(1, np.arange(2))
            * np.where(np.arange(2) % 2 > 0, -1, 1)
            * beta(self.a + 0.5 - np.arange(2), self.b - 0.5 + np.arange(2))
        ).sum()
        nu2 = ((self.a + self.b) / (4 * beta(self.a, self.b))) * (
            comb(2, np.arange(3))
            * np.where(np.arange(3) % 2 > 0, -1, 1)
            * beta(self.a + 1 - np.arange(3), self.b - 1 + np.arange(3))
        ).sum() - nu1**2
        nu3 = (
            ((self.a + self.b) ** 1.5 / (8 * beta(self.a, self.b)))
            * (
                comb(3, np.arange(4))
                * np.where(np.arange(4) % 2 > 0, -1, 1)
                * beta(self.a + 1.5 - np.arange(4), self.b - 1.5 + np.arange(4))
            ).sum()
            - nu1**3
            - 3 * nu1 * nu2
        )
        nu4 = (
            ((self.a + self.b) ** 2 / (16 * beta(self.a, self.b)))
            * (
                comb(4, np.arange(5))
                * np.where(np.arange(5) % 2 > 0, -1, 1)
                * beta(self.a + 2 - np.arange(5), self.b - 2 + np.arange(5))
            ).sum()
            - 4 * nu3 * nu1
            - nu1**4
            - 6 * nu2 * nu1**2
        )
        return nu4 / nu2**2 - 3

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        beta_rng = random_state.beta(self.a, self.b, size)
        return ((2 * beta_rng - 1) * np.sqrt(self.a + self.b)) / (
            2 * np.sqrt(beta_rng * (1 - beta_rng))
        ) * self.sigma + self.mu

    def _fit_moments(self, mean, sigma):
        optimize_moments(self, mean, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@nb.vectorize(nopython=True, cache=True)
def nb_cdf(x, mu, sigma, a, b, lower, upper):
    x = (x - mu) / sigma
    if x == -np.inf:
        return 0
    return cdf_bounds(betainc(a, b, (1 + x / np.sqrt(a + b + x**2)) * 0.5), x, lower, upper)


@nb.njit(cache=True)
def nb_ppf(q, mu, sigma, a, b, lower, upper):
    x_val = betaincinv(a, b, q)
    return ppf_bounds_cont(
        ((2 * x_val - 1) * np.sqrt(a + b)) / (2 * np.sqrt(x_val * (1 - x_val))) * sigma + mu,
        q,
        lower,
        upper,
    )


@nb.njit(cache=True)
def nb_logpdf(x, mu, sigma, a, b):
    x = (x - mu) / sigma
    return np.log(
        (1 + x / np.sqrt(a + b + x**2)) ** (a + 0.5)
        * (1 - x / np.sqrt(a + b + x**2)) ** (b + 0.5)
        / (2 ** (a + b - 1) * beta(a, b) * np.sqrt(a + b) * sigma)
    )


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, sigma, a, b):
    return -(nb_logpdf(x, mu, sigma, a, b)).sum()
