import numpy as np
from pytensor_distributions import wald as ptd_wald

from preliz.distributions.distributions import Continuous
from preliz.internal.distribution_helper import all_not_none, eps, pytensor_jit, pytensor_rng_jit
from preliz.internal.optimization import optimize_ml


class Wald(Continuous):
    r"""
    Wald distribution.

    The pdf of this distribution is

    .. math::

       f(x \mid \mu, \lambda) =
           \left(\frac{\lambda}{2\pi}\right)^{1/2} x^{-3/2}
           \exp\left\{
               -\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2
           \right\}

    .. plot::
        :context: close-figs


        from preliz import Wald, style
        style.use('preliz-doc')
        mus = [1., 1.]
        lams = [1., 3.]
        for mu, lam in zip(mus, lams):
            Wald(mu, lam).plot_pdf(support=(0,4))

    ========  =============================
    Support   :math:`x \in (0, \infty)`
    Mean      :math:`\mu`
    Variance  :math:`\dfrac{\mu^3}{\lambda}`
    ========  =============================

    Wald distribution has 3 alternative parametrizations. In terms of mu and lam,
    mu and phi or lam and phi.

    The link between the 3 alternatives is given by

    .. math::

       \phi = \dfrac{\lambda}{\mu}

    Parameters
    ----------
    mu : float
        Mean of the distribution (mu > 0).
    lam : float
        Relative precision (lam > 0).
    phi : float
        Shape parameter (phi > 0).
    """

    def __init__(self, mu=None, lam=None, phi=None):
        super().__init__()
        self.support = (0, np.inf)
        self._parametrization(mu, lam, phi)

    def _parametrization(self, mu=None, lam=None, phi=None):
        if all_not_none(mu, lam, phi):
            raise ValueError(
                "Incompatible parametrization. Either use mu and lam or mu and phi or lam and phi."
            )

        self.param_names = ("mu", "lam")
        self.params_support = ((eps, np.inf), (eps, np.inf))

        if phi is not None:
            self.phi = phi
            if (mu and phi) is not None:
                lam = self._from_mu_phi(mu, phi)
                self.param_names = ("mu", "phi")

            elif (lam and phi) is not None:
                mu = self._from_lam_phi(lam, phi)
                self.param_names = ("lam", "phi")

        self.mu = mu
        self.lam = lam
        if all_not_none(self.mu, self.lam):
            self._update(self.mu, self.lam)

    def _from_mu_phi(self, mu, phi):
        lam = mu * phi
        return lam

    def _from_lam_phi(self, lam, phi):
        mu = lam / phi
        return mu

    def _to_phi(self, mu, lam):
        phi = lam / mu
        return phi

    def _update(self, mu, lam):
        self.mu = np.float64(mu)
        self.lam = np.float64(lam)
        self.phi = self._to_phi(self.mu, self.lam)

        if self.param_names == ("mu", "lam"):
            self.params = (self.mu, self.lam)
        elif self.param_names == ("mu", "phi"):
            self.params = (self.mu, self.phi)
        elif self.param_names == ("lam", "phi"):
            self.params = (self.lam, self.phi)

        self.is_frozen = True

    def pdf(self, x):
        return ptd_pdf(x, self.mu, self.lam)

    def cdf(self, x):
        return ptd_cdf(x, self.mu, self.lam)

    def ppf(self, q):
        return ptd_ppf(q, self.mu, self.lam)

    def logpdf(self, x):
        return ptd_logpdf(x, self.mu, self.lam)

    def entropy(self):
        return ptd_entropy(self.mu, self.lam)

    def mean(self):
        return ptd_mean(self.mu, self.lam)

    def mode(self):
        return ptd_mode(self.mu, self.lam)

    def median(self):
        return ptd_median(self.mu, self.lam)

    def var(self):
        return ptd_var(self.mu, self.lam)

    def std(self):
        return ptd_std(self.mu, self.lam)

    def skewness(self):
        return ptd_skewness(self.mu, self.lam)

    def kurtosis(self):
        return ptd_kurtosis(self.mu, self.lam)

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        return ptd_rvs(self.mu, self.lam, size=size, rng=random_state)

    def _fit_moments(self, mean, sigma):
        lam = mean**3 / sigma**2
        self._update(mean, lam)

    def _fit_mle(self, sample):
        optimize_ml(self, sample)


@pytensor_jit
def ptd_pdf(x, mu, lam):
    return ptd_wald.pdf(x, mu, lam)


@pytensor_jit
def ptd_cdf(x, mu, lam):
    return ptd_wald.cdf(x, mu, lam)


@pytensor_jit
def ptd_ppf(q, mu, lam):
    return ptd_wald.ppf(q, mu, lam)


@pytensor_jit
def ptd_logpdf(x, mu, lam):
    return ptd_wald.logpdf(x, mu, lam)


@pytensor_jit
def ptd_entropy(mu, lam):
    return ptd_wald.entropy(mu, lam)


@pytensor_jit
def ptd_mean(mu, lam):
    return ptd_wald.mean(mu, lam)


@pytensor_jit
def ptd_mode(mu, lam):
    return ptd_wald.mode(mu, lam)


@pytensor_jit
def ptd_median(mu, lam):
    return ptd_wald.median(mu, lam)


@pytensor_jit
def ptd_var(mu, lam):
    return ptd_wald.var(mu, lam)


@pytensor_jit
def ptd_std(mu, lam):
    return ptd_wald.std(mu, lam)


@pytensor_jit
def ptd_skewness(mu, lam):
    return ptd_wald.skewness(mu, lam)


@pytensor_jit
def ptd_kurtosis(mu, lam):
    return ptd_wald.kurtosis(mu, lam)


@pytensor_rng_jit
def ptd_rvs(mu, lam, size, rng):
    return ptd_wald.rvs(mu, lam, size=size, random_state=rng)
