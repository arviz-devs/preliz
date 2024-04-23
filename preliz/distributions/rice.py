# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np

from scipy.special import i0, i1, i0e, chndtr, chndtrix  # pylint: disable=no-name-in-module
from .distributions import Continuous
from ..internal.optimization import optimize_moments_rice, optimize_ml
from ..internal.distribution_helper import eps, all_not_none
from ..internal.special import ppf_bounds_cont, cdf_bounds


class Rice(Continuous):
    r"""
    Rice distribution.

    The pdf of this distribution is

    .. math::

        f(x\mid \nu ,\sigma )=
            {\frac  {x}{\sigma ^{2}}}\exp
            \left({\frac  {-(x^{2}+\nu ^{2})}
            {2\sigma ^{2}}}\right)I_{0}\left({\frac  {x\nu }{\sigma ^{2}}}\right)

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import Rice
        az.style.use('arviz-doc')
        nus = [0., 0., 4.]
        sigmas = [1., 2., 2.]
        for nu, sigma in  zip(nus, sigmas):
            Rice(nu, sigma).plot_pdf(support=(0,10))

    ========  ==============================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\sigma {\sqrt  {\pi /2}}\,\,L_{{1/2}}(-\nu ^{2}/2\sigma ^{2})`
    Variance  :math:`2\sigma ^{2}+\nu ^{2}-{\frac  {\pi \sigma ^{2}}{2}}L_{{1/2}}^{2}
                        \left({\frac  {-\nu ^{2}}{2\sigma ^{2}}}\right)`
    ========  ==============================================================

    Rice distribution has 2 alternative parameterizations. In terms of nu and sigma
    or b and sigma.

    The link between the two parametrizations is given by

    .. math::

       b = \dfrac{\nu}{\sigma}

    Parameters
    ----------
    nu : float
        Noncentrality parameter.
    sigma : float
        Scale parameter.
    b : float
        Shape parameter.
    """

    def __init__(self, nu=None, sigma=None, b=None):
        super().__init__()
        self.name = "rice"
        self.support = (0, np.inf)
        self._parametrization(nu, sigma, b)

    def _parametrization(self, nu=None, sigma=None, b=None):
        if all_not_none(nu, b):
            raise ValueError(
                "Incompatible parametrization. Either use nu and sigma or b and sigma."
            )

        self.param_names = ("nu", "sigma")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if b is not None:
            self.b = b
            self.sigma = sigma
            self.param_names = ("b", "sigma")
            if all_not_none(b, sigma):
                nu = self._from_b(b, sigma)

        self.nu = nu
        self.sigma = sigma
        if all_not_none(self.nu, self.sigma):
            self._update(self.nu, self.sigma)

    def _from_b(self, b, sigma):
        nu = b * sigma
        return nu

    def _to_b(self, nu, sigma):
        b = nu / sigma
        return b

    def _update(self, nu, sigma):
        self.nu = np.float64(nu)
        self.sigma = np.float64(sigma)
        self.b = self._to_b(self.nu, self.sigma)

        if self.param_names[0] == "nu":
            self.params = (self.nu, self.sigma)
        elif self.param_names[0] == "b":
            self.params = (self.b, self.sigma)

        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        return np.exp(self.logpdf(x))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.nu, self.sigma)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.nu, self.sigma)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.nu, self.sigma)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.nu, self.sigma)

    def entropy(self):
        x_values = self.xvals("restricted")
        logpdf = self.logpdf(x_values)
        return -np.trapz(np.exp(logpdf) * logpdf, x_values)

    def mean(self):
        return self.sigma * np.sqrt(np.pi / 2) * _l_half(-self.nu**2 / (2 * self.sigma**2))

    def median(self):
        return self.ppf(0.5)

    def var(self):
        return (
            2 * self.sigma**2
            + self.nu**2
            - np.pi / 2 * self.sigma**2 * _l_half(-self.nu**2 / (2 * self.sigma**2)) ** 2
        )

    def std(self):
        return self.var() ** 2

    def skewness(self):
        return NotImplemented

    def kurtosis(self):
        return NotImplemented

    def rvs(self, size=1, random_state=None):
        random_state = np.random.default_rng(random_state)
        t_v = (self.nu / self.sigma) / np.sqrt(2) + random_state.standard_normal(size=(2, size))
        return np.sqrt((t_v * t_v).sum(axis=0)) * self.sigma

    def _fit_moments(self, mean, sigma):
        nu, sigma = optimize_moments_rice(mean, sigma)
        self._update(nu, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


def nb_cdf(x, nu, sigma):
    return cdf_bounds(chndtr((x / sigma) ** 2, 2, (nu / sigma) ** 2), x, 0, np.inf)


def nb_ppf(q, nu, sigma):
    return ppf_bounds_cont(np.sqrt(chndtrix(q, 2, (nu / sigma) ** 2)) * sigma, q, 0, np.inf)


def nb_logpdf(x, nu, sigma):
    b = nu / sigma
    x = x / sigma
    return np.where(
        x < 0, -np.inf, np.log(x * np.exp((-(x - b) * (x - b)) / 2) * i0e(x * b) / sigma)
    )


def nb_neg_logpdf(x, nu, sigma):
    return -(nb_logpdf(x, nu, sigma)).sum()


def _l_half(x):
    return np.exp(x / 2) * ((1 - x) * i0(-x / 2) - x * i1(-x / 2))
