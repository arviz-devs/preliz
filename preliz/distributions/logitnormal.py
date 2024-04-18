# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numba as nb
import numpy as np

from .distributions import Continuous
from ..internal.distribution_helper import eps, to_precision, from_precision, all_not_none
from ..internal.special import erf, erfinv, logit, expit, mean_and_std, cdf_bounds, ppf_bounds_cont


class LogitNormal(Continuous):
    r"""
    Logit-Normal distribution.

    The pdf of this distribution is

    .. math::
       f(x \mid \mu, \tau) =
           \frac{1}{x(1-x)} \sqrt{\frac{\tau}{2\pi}}
           \exp\left\{ -\frac{\tau}{2} (logit(x)-\mu)^2 \right\}


    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import LogitNormal
        az.style.use('arviz-doc')
        mus = [0., 0., 0., 1.]
        sigmas = [0.3, 1., 2., 1.]
        for mu, sigma in zip(mus, sigmas):
            LogitNormal(mu, sigma).plot_pdf()

    ========  ==========================================
    Support   :math:`x \in (0, 1)`
    Mean      no analytical solution
    Variance  no analytical solution
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Location parameter.
    sigma : float
        Scale parameter (sigma > 0).
    tau : float
        Scale parameter (tau > 0).
    """

    def __init__(self, mu=None, sigma=None, tau=None):
        super().__init__()
        self.support = (0, 1)
        self._parametrization(mu, sigma, tau)

    def _parametrization(self, mu=None, sigma=None, tau=None):
        if all_not_none(sigma, tau):
            raise ValueError(
                "Incompatible parametrization. Either use mu and sigma, or mu and tau."
            )

        names = ("mu", "sigma")
        self.params_support = ((-np.inf, np.inf), (eps, np.inf))

        if tau is not None:
            self.tau = tau
            sigma = from_precision(tau)
            names = ("mu", "tau")

        self.mu = mu
        self.sigma = sigma
        self.param_names = names
        if all_not_none(mu, sigma):
            self._update(mu, sigma)

    def _get_frozen(self):
        frozen = None
        if all_not_none(self.params):
            frozen = self.dist(self.mu, self.sigma)
        return frozen

    def _update(self, mu, sigma):
        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.tau = to_precision(sigma)

        if self.param_names[1] == "sigma":
            self.params = (self.mu, self.sigma)
        elif self.param_names[1] == "tau":
            self.params = (self.mu, self.tau)

        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.mu, self.sigma)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.mu, self.sigma)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu, self.sigma)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.sigma)

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        return -np.trapz(np.exp(logpdf) * logpdf, x_values)

    def mean(self):
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        return np.trapz(x_values * pdf, x_values)

    def median(self):
        return self.ppf(0.5)

    def var(self):
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        return np.trapz((x_values - self.mean()) ** 2 * pdf, x_values)

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        mean = self.mean()
        std = self.std()
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        return np.trapz(((x_values - mean) / std) ** 3 * pdf, x_values)

    def kurtosis(self):
        mean = self.mean()
        std = self.std()
        x_values = self.xvals("full")
        pdf = self.pdf(x_values)
        return np.trapz(((x_values - mean) / std) ** 4 * pdf, x_values) - 3

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return expit(random_state.normal(self.mu, self.sigma, size))

    def _fit_moments(self, mean, sigma):
        mu = logit(mean)
        sigma = np.diff((mean - sigma * 3, mean + sigma * 3))
        self._update(mu, sigma)

    def _fit_mle(self, sample):
        mu, sigma = mean_and_std(logit(sample))
        self._update(mu, sigma)


@nb.njit(cache=True)
def nb_cdf(x, mu, sigma):
    return cdf_bounds(0.5 * (1 + erf((logit(x) - mu) / (sigma * 2**0.5))), x, 0, 1)


@nb.njit(cache=True)
def nb_ppf(q, mu, sigma):
    return ppf_bounds_cont(expit(mu + sigma * 2**0.5 * erfinv(2 * q - 1)), q, 0, 1)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, mu, sigma):
    if x <= 0:
        return -np.inf
    if x >= 1:
        return -np.inf
    else:
        return (
            -np.log(sigma)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * ((logit(x) - mu) / sigma) ** 2
            - np.log(x)
            - np.log1p(-x)
        )


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, sigma):
    return -(nb_logpdf(x, mu, sigma)).sum()
