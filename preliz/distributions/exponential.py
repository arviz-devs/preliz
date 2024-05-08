# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from .distributions import Continuous
from ..internal.distribution_helper import eps, all_not_none
from ..internal.special import cdf_bounds, ppf_bounds_cont, mean_sample, xlog1py


class Exponential(Continuous):
    r"""
    Exponential Distribution

    The pdf of this distribution is

    .. math::

        f(x \mid \lambda) = \lambda \exp\left\{ -\lambda x \right\}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Exponential
        az.style.use('arviz-doc')
        for lam in [0.5,  2.]:
            Exponential(lam).plot_pdf(support=(0,5))


    ========  ============================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{1}{\lambda}`
    Variance  :math:`\dfrac{1}{\lambda^2}`
    ========  ============================

    Exponential distribution has 2 alternative parametrizations. In terms of lambda (rate)
    or in terms of beta (scale).

    The link between the two alternatives is given by:

    .. math::

        \beta = \dfrac{1}{\lambda}

    Parameters
    ----------
    lam : float
        Rate or inverse scale (lam > 0).
    beta : float
        Scale (beta > 0).
    """

    def __init__(self, lam=None, beta=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(lam, beta)

    def _parametrization(self, lam=None, beta=None):
        if all_not_none(lam, beta):
            raise ValueError("Incompatible parametrization. Either use 'lam' or 'beta'.")

        self.param_names = ("lam",)
        self.params_support = ((eps, np.inf),)

        if beta is not None:
            lam = 1 / beta
            self.param_names = ("beta",)

        self.lam = lam
        self.beta = beta
        if self.lam is not None:
            self._update(self.lam)

    def _update(self, lam):
        self.lam = np.float64(lam)
        self.beta = 1 / self.lam

        if self.param_names[0] == "lam":
            self.params = (self.lam,)
        elif self.param_names[0] == "beta":
            self.params = (self.beta,)

        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.lam))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.lam)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.beta)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.lam)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.lam)

    def entropy(self):
        return nb_entropy(self.beta)

    def median(self):
        return np.log(2) * self.beta

    def mean(self):
        return self.beta

    def std(self):
        return self.beta

    def var(self):
        return self.beta**2

    def skewness(self):
        return 2

    def kurtosis(self):
        return 6

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return random_state.exponential(self.beta, size)

    def _fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        lam = 1 / mean
        self._update(lam)

    def _fit_mle(self, sample, **kwargs):
        mean = mean_sample(sample)
        self._update(1 / mean)


@nb.njit(cache=True)
def nb_cdf(x, lam):
    x_lam = lam * x
    return cdf_bounds(1 - np.exp(-x_lam), x, 0, np.inf)


@nb.njit(cache=True)
def nb_ppf(q, beta):
    return ppf_bounds_cont(-xlog1py(beta, -q), q, 0, np.inf)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, lam):
    if x < 0:
        return -np.inf
    else:
        return np.log(lam) - lam * x


@nb.njit(cache=True)
def nb_neg_logpdf(x, lam):
    return (-nb_logpdf(x, lam)).sum()


@nb.njit(cache=True)
def nb_entropy(beta):
    return 1 + np.log(beta)
