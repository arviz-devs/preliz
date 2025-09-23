import numpy as np
from pytensor_distributions import rice as ptd_rice

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml, optimize_moments_rice


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


        from preliz import Rice, style
        style.use('preliz-doc')
        nus = [0., 0., 4.]
        sigmas = [1., 2., 2.]
        for nu, sigma in  zip(nus, sigmas):
            Rice(nu, sigma).plot_pdf(support=(0,10))

    ========  ==============================================================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\sigma \sqrt{\pi /2} L_{1/2}(-\nu^2 / 2\sigma^2)`
    Variance  :math:`2\sigma^2 + \nu^2 - \frac{\pi \sigma^2}{2}`
              :math:`L_{1/2}^2\left(\frac{-\nu^2}{2\sigma^2}\right)`
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
        return ptd_pdf(x, self.nu, self.sigma)

    def cdf(self, x):
        return ptd_cdf(x, self.nu, self.sigma)

    def ppf(self, q):
        return ptd_ppf(q, self.nu, self.sigma)

    def logpdf(self, x):
        return ptd_logpdf(x, self.nu, self.sigma)

    def entropy(self):
        return ptd_entropy(self.nu, self.sigma)

    def mean(self):
        return ptd_mean(self.nu, self.sigma)

    def median(self):
        return ptd_median(self.nu, self.sigma)

    def var(self):
        return ptd_var(self.nu, self.sigma)

    def std(self):
        return ptd_std(self.nu, self.sigma)

    def skewness(self):
        return ptd_skewness(self.nu, self.sigma)

    def kurtosis(self):
        return ptd_kurtosis(self.nu, self.sigma)

    def rvs(self, size=1, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.nu, self.sigma, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        nu, sigma = optimize_moments_rice(mean, sigma)
        self._update(nu, sigma)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, nu, sigma):
    return ptd_rice.pdf(x, nu, sigma)


@pytensor_jit
def ptd_cdf(x, nu, sigma):
    return ptd_rice.cdf(x, nu, sigma)


@pytensor_jit
def ptd_ppf(q, nu, sigma):
    return ptd_rice.ppf(q, nu, sigma)


@pytensor_jit
def ptd_logpdf(x, nu, sigma):
    return ptd_rice.logpdf(x, nu, sigma)


@pytensor_jit
def ptd_entropy(nu, sigma):
    return ptd_rice.entropy(nu, sigma)


@pytensor_jit
def ptd_mean(nu, sigma):
    return ptd_rice.mean(nu, sigma)


@pytensor_jit
def ptd_mode(nu, sigma):
    return ptd_rice.mode(nu, sigma)


@pytensor_jit
def ptd_median(nu, sigma):
    return ptd_rice.median(nu, sigma)


@pytensor_jit
def ptd_var(nu, sigma):
    return ptd_rice.var(nu, sigma)


@pytensor_jit
def ptd_std(nu, sigma):
    return ptd_rice.std(nu, sigma)


@pytensor_jit
def ptd_skewness(nu, sigma):
    return ptd_rice.skewness(nu, sigma)


@pytensor_jit
def ptd_kurtosis(nu, sigma):
    return ptd_rice.kurtosis(nu, sigma)


@pytensor_rng_jit
def ptd_rvs(nu, sigma, size, rng):
    return ptd_rice.rvs(nu, sigma, size=size, random_state=rng)
