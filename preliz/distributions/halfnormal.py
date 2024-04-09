# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import eps, to_precision, from_precision, all_not_none
from ..internal.special import half_erf, erfinv, ppf_bounds_cont


class HalfNormal(Continuous):
    r"""
    HalfNormal Distribution

    The pdf of this distribution is

    .. math::

       f(x \mid \sigma) =
           \sqrt{\frac{2}{\pi\sigma^2}}
           \exp\left(\frac{-x^2}{2\sigma^2}\right)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import HalfNormal
        az.style.use('arviz-doc')
        for sigma in [0.4,  2.]:
            HalfNormal(sigma).plot_pdf(support=(0,5))

    ========  ==========================================
    Support   :math:`x \in [0, \infty)`
    Mean      :math:`\dfrac{\sigma \sqrt{2}}{\sqrt{\pi}}`
    Variance  :math:`\sigma^2\left(1 - \dfrac{2}{\pi}\right)`
    ========  ==========================================

    HalfNormal distribution has 2 alternative parameterizations. In terms of sigma (standard
    deviation) or tau (precision).

    The link between the 2 alternatives is given by

    .. math::

        \tau = \frac{1}{\sigma^2}

    Parameters
    ----------
    sigma : float
        Scale parameter :math:`\sigma` (``sigma`` > 0).
    tau : float
        Precision :math:`\tau` (``tau`` > 0).
    """

    def __init__(self, sigma=None, tau=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(sigma, tau)

    def _parametrization(self, sigma=None, tau=None):
        if all_not_none(sigma, tau):
            raise ValueError("Incompatible parametrization. Either use sigma or tau.")

        self.param_names = ("sigma",)
        self.params_support = ((eps, np.inf),)

        if tau is not None:
            sigma = from_precision(tau)
            self.param_names = ("tau",)

        self.sigma = sigma
        self.tau = tau
        if self.sigma is not None:
            self._update(self.sigma)

    def _update(self, sigma):
        self.sigma = np.float64(sigma)
        self.tau = to_precision(self.sigma)

        if self.param_names[0] == "sigma":
            self.params = (self.sigma,)
        elif self.param_names[0] == "tau":
            self.params = (self.tau,)

        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.sigma))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.sigma)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.sigma, self.support[0], self.support[1])

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.sigma)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.sigma)

    def entropy(self):
        return nb_entropy(self.sigma)

    def mean(self):
        return self.sigma * 0.7978845608028655

    def median(self):
        return self.sigma * 0.6744897501960818

    def var(self):
        return self.sigma**2 * 0.3633802276324186

    def std(self):
        return self.sigma * 0.6028102749890869

    def skewness(self):
        return 0.9952717464311565

    def kurtosis(self):
        return 0.8691773036059736

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return np.abs(random_state.normal(0, self.sigma, size))

    def _fit_moments(self, mean, sigma):  # pylint: disable=unused-argument
        self._update(sigma / (1 - 2 / np.pi) ** 0.5)

    def _fit_mle(self, sample):
        self._update(nb_fit_mle(sample))


@nb.njit(cache=True)
def nb_cdf(x, sigma):
    return half_erf(x / (sigma * 2**0.5))


@nb.njit(cache=True)
def nb_ppf(q, sigma, lower, upper):
    x_vals = np.asarray(sigma * 2**0.5 * erfinv(q))
    return ppf_bounds_cont(x_vals, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(sigma):
    return 0.5 * np.log(np.pi * sigma**2.0 / 2.0) + 0.5


@nb.njit(cache=True)
def nb_fit_mle(sample):
    return np.mean(sample**2) ** 0.5


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, sigma):
    if x < 0:
        return -np.inf
    else:
        return np.log(np.sqrt(2 / np.pi)) + np.log(1 / sigma) - 0.5 * ((x / sigma) ** 2)


@nb.njit(cache=True)
def nb_neg_logpdf(x, sigma):
    return -(nb_logpdf(x, sigma)).sum()
