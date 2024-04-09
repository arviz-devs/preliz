# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from .distributions import Continuous
from ..internal.distribution_helper import all_not_none, eps


class AsymmetricLaplace(Continuous):
    r"""
    Asymmetric-Laplace distribution.

    The pdf of this distribution is

    .. math::
        {f(x|\\b,\kappa,\mu) =
            \left({\frac{\\b}{\kappa + 1/\kappa}}\right)\,e^{-(x-\mu)\\b\,s\kappa ^{s}}}

    where

    .. math::

        s = sgn(x-\mu)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import AsymmetricLaplace
        az.style.use('arviz-doc')
        kappas = [1., 2., .5]
        mus = [0., 0., 3.]
        bs = [1., 1., 1.]
        for kappa, mu, b in zip(kappas, mus, bs):
            AsymmetricLaplace(kappa, mu, b).plot_pdf(support=(-10,10))

    ========  ========================
    Support   :math:`x \in \mathbb{R}`
    Mean      :math:`\mu-\frac{\\\kappa-1/\kappa}b`
    Variance  :math:`\frac{1+\kappa^{4}}{b^2\kappa^2 }`
    ========  ========================

    AsymmetricLaplace distribution has 2 alternative parametrizations. In terms of kappa,
    mu and b or q, mu and b.

    The link between the 2 alternatives is given by

    .. math::

       \kappa = \sqrt(\frac{q}{1-q})

    Parameters
    ----------
    kappa : float
        Symmetry parameter (kappa > 0).
    mu : float
        Location parameter.
    b : float
        Scale parameter (b > 0).
    q : float
        Symmetry parameter (0 < q < 1).
    """

    def __init__(self, kappa=None, mu=None, b=None, q=None):
        super().__init__()
        self.support = (-np.inf, np.inf)
        self._parametrization(kappa, mu, b, q)

    def _parametrization(self, kappa=None, mu=None, b=None, q=None):
        if all_not_none(kappa, q):
            raise ValueError("Incompatible parametrization. Either use kappa or q.")

        self.param_names = ("kappa", "mu", "b")
        self.params_support = ((eps, np.inf), (-np.inf, np.inf), (eps, np.inf))

        if q is not None:
            self.q = q
            kappa = self._from_q(q)
            self.param_names = ("q", "mu", "b")
            self.params_support = ((eps, 1 - eps), (-np.inf, np.inf), (eps, np.inf))

        self.kappa = kappa
        self.mu = mu
        self.b = b
        if all_not_none(kappa, mu, b):
            self._update(kappa, mu, b)

    def _from_q(self, q):
        kappa = (q / (1 - q)) ** 0.5
        return kappa

    def _to_q(self, kappa):
        q = kappa**2 / (1 + kappa**2)
        return q

    def _update(self, kappa, mu, b):
        self.kappa = np.float64(kappa)
        self.mu = np.float64(mu)
        self.b = np.float64(b)
        self.q = self._to_q(self.kappa)

        if self.param_names[0] == "kappa":
            self.params = (self.kappa, self.mu, self.b)
        elif self.param_names[0] == "q":
            self.params = (self.q, self.mu, self.b)

        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.mu, self.b, self.kappa))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.mu, self.b, self.kappa)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.mu, self.b, self.kappa)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu, self.b, self.kappa)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.b, self.kappa)

    def entropy(self):
        return nb_entropy(self.b, self.kappa)

    def median(self):
        if self.kappa > 1:
            return self.mu + self.kappa * self.b * np.log(
                (1 + self.kappa**2) / (2 * self.kappa**2)
            )
        return self.mu - np.log((1 + self.kappa**2) / 2) / (self.kappa / self.b)

    def mean(self):
        return (1 / self.kappa - self.kappa) * self.b + self.mu

    def var(self):
        return ((1 / self.kappa) ** 2 + self.kappa**2) * self.b**2

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        return 2.0 * (1 - np.power(self.kappa, 6)) / np.power(1 + np.power(self.kappa, 4), 1.5)

    def kurtosis(self):
        return 6.0 * (1 + np.power(self.kappa, 8)) / np.power(1 + np.power(self.kappa, 4), 2)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        random_samples = random_state.uniform(
            -self.kappa, 1 / self.kappa, size  # pylint: disable=invalid-unary-operand-type
        )
        return nb_rvs(random_samples, self.mu, self.b, self.kappa)

    def _fit_moments(self, mean, sigma):
        # Assume symmetry
        mu = mean
        b = (sigma / 2) * (2**0.5)
        self._update(1, mu, b)

    def _fit_mle(self, sample, **kwargs):
        kappa, mu, b = nb_fit_mle(sample)
        self._update(kappa, mu, b)


@nb.vectorize(nopython=True, cache=True)
def nb_cdf(x, mu, b, kappa):
    x = (x - mu) / b
    kap_inv = 1 / kappa
    kap_kapinv = kappa + kap_inv
    if x >= 0:
        return 1 - np.exp(-x * kappa) * (kap_inv / kap_kapinv)
    return np.exp(x * kap_inv) * (kappa / kap_kapinv)


@nb.vectorize(nopython=True, cache=True)
def nb_ppf(q, mu, b, kappa):
    kap_inv = 1 / kappa
    kap_kapinv = kappa + kap_inv
    if q >= kappa / kap_kapinv:
        q_ppf = -np.log((1 - q) * kap_kapinv * kappa) * kap_inv
    else:
        q_ppf = np.log(q * kap_kapinv / kappa) * kappa
    return q_ppf * b + mu


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, mu, b, kappa):
    x = (x - mu) / b
    kap_inv = 1 / kappa
    if x >= 0:
        ald_x = x * -kappa
    else:
        ald_x = x * kap_inv
    ald_x -= np.log(kappa + kap_inv)
    return ald_x - np.log(b)


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, b, kappa):
    return (-nb_logpdf(x, mu, b, kappa)).sum()


@nb.njit(cache=True)
def nb_rvs(random_samples, mu, b, kappa):
    sgn = np.sign(random_samples)
    return mu - (1 / (1 / b * sgn * kappa**sgn)) * np.log(1 - random_samples * sgn * kappa**sgn)


@nb.njit(cache=True)
def nb_entropy(b, kappa):
    return 1 + np.log(kappa + 1 / kappa) + np.log(b)


@nb.njit(cache=True)
def nb_fit_mle(sample):
    new_mu = np.median(sample)
    new_b = np.mean(np.abs(sample - new_mu))
    new_kappa = np.sum((sample - new_mu) * np.sign(sample - new_mu)) / np.sum(
        np.abs(sample - new_mu)
    )
    return new_kappa, new_mu, new_b
