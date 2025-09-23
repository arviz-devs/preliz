import numpy as np
import pytensor.tensor as pt
from pytensor_distributions import asymmetriclaplace as ptd_asymmetriclaplace

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


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


        from preliz import AsymmetricLaplace, style
        style.use('preliz-doc')
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
        self.support = (-pt.inf, pt.inf)
        self._parametrization(kappa, mu, b, q)

    def _parametrization(self, kappa=None, mu=None, b=None, q=None):
        if all_not_none(kappa, q):
            raise ValueError("Incompatible parametrization. Either use kappa or q.")

        self.param_names = ("kappa", "mu", "b")
        self.params_support = ((eps, pt.inf), (-pt.inf, pt.inf), (eps, pt.inf))

        if q is not None:
            self.q = q
            kappa = ptd_asymmetriclaplace.from_q(q)
            self.param_names = ("q", "mu", "b")
            self.params_support = ((eps, 1 - eps), (-pt.inf, pt.inf), (eps, pt.inf))

        self.kappa = kappa
        self.mu = mu
        self.b = b
        if all_not_none(kappa, mu, b):
            self._update(kappa, mu, b)

    def _update(self, kappa, mu, b):
        self.kappa = np.float64(kappa)
        self.mu = np.float64(mu)
        self.b = np.float64(b)
        self.q = ptd_asymmetriclaplace.to_q(self.kappa)

        if self.param_names[0] == "kappa":
            self.params = (self.kappa, self.mu, self.b)
        elif self.param_names[0] == "q":
            self.params = (self.q, self.mu, self.b)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.b, self.kappa)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.b, self.kappa)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.b, self.kappa)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.b, self.kappa)

    def entropy(self):
        return ptd_entropy(self.mu, self.b, self.kappa)

    def median(self):
        return ptd_median(self.mu, self.b, self.kappa)

    def mean(self):
        return ptd_mean(self.mu, self.b, self.kappa)

    def mode(self):
        return ptd_mode(self.mu, self.b, self.kappa)

    def var(self):
        return ptd_var(self.mu, self.b, self.kappa)

    def std(self):
        return ptd_std(self.mu, self.b, self.kappa)

    def skewness(self):
        return ptd_skewness(self.mu, self.b, self.kappa)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.b, self.kappa)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.b, self.kappa, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        # Assume symmetry
        mu = mean
        b = (sigma / 2) * (2**0.5)
        self._update(1, mu, b)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, mu, b, kappa):
    return ptd_asymmetriclaplace.pdf(x, mu, b, kappa)


@pytensor_jit
def ptd_cdf(x, mu, b, kappa):
    return ptd_asymmetriclaplace.cdf(x, mu, b, kappa)


@pytensor_jit
def ptd_ppf(q, mu, b, kappa):
    return ptd_asymmetriclaplace.ppf(q, mu, b, kappa)


@pytensor_jit
def ptd_logpdf(x, mu, b, kappa):
    return ptd_asymmetriclaplace.logpdf(x, mu, b, kappa)


@pytensor_jit
def ptd_entropy(mu, b, kappa):
    return ptd_asymmetriclaplace.entropy(mu, b, kappa)


@pytensor_jit
def ptd_mean(mu, b, kappa):
    return ptd_asymmetriclaplace.mean(mu, b, kappa)


@pytensor_jit
def ptd_mode(mu, b, kappa):
    return ptd_asymmetriclaplace.mode(mu, b, kappa)


@pytensor_jit
def ptd_median(mu, b, kappa):
    return ptd_asymmetriclaplace.median(mu, b, kappa)


@pytensor_jit
def ptd_var(mu, b, kappa):
    return ptd_asymmetriclaplace.var(mu, b, kappa)


@pytensor_jit
def ptd_std(mu, b, kappa):
    return ptd_asymmetriclaplace.std(mu, b, kappa)


@pytensor_jit
def ptd_skewness(mu, b, kappa):
    return ptd_asymmetriclaplace.skewness(mu, b, kappa)


@pytensor_jit
def ptd_kurtosis(mu, b, kappa):
    return ptd_asymmetriclaplace.kurtosis(mu, b, kappa)


@pytensor_rng_jit
def ptd_rvs(mu, b, kappa, size, rng):
    return ptd_asymmetriclaplace.rvs(mu, b, kappa, size=size, random_state=rng)
