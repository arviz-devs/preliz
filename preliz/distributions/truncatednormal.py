# pylint: disable=attribute-defined-outside-init
# pylint: disable=arguments-differ
import numpy as np
import numba as nb

from ..internal.special import cdf_bounds, erf, erfinv, ppf_bounds_cont
from ..internal.optimization import optimize_ml
from ..internal.distribution_helper import eps, all_not_none
from .distributions import Continuous


class TruncatedNormal(Continuous):
    r"""
    TruncatedNormal distribution.

    The pdf of this distribution is

    .. math::

       f(x;\mu ,\sigma ,a,b)={\frac {\phi ({\frac {x-\mu }{\sigma }})}{
            \sigma \left(\Phi ({\frac {b-\mu }{\sigma }})-\Phi ({\frac {a-\mu }{\sigma }})\right)}}

    .. plot::
        :context: close-figs

        import arviz as az
        from preliz import TruncatedNormal
        az.style.use('arviz-doc')
        mus = [0.,  0., 0.]
        sigmas = [3.,5.,7.]
        lowers = [-3, -5, -5]
        uppers = [7, 5, 4]
        for mu, sigma, lower, upper in zip(mus, sigmas,lowers,uppers):
            TruncatedNormal(mu, sigma, lower, upper).plot_pdf(support=(-10,10))

    ========  ==========================================
    Support   :math:`x \in [a, b]`
    Mean      :math:`\mu +{\frac {\phi (\alpha )-\phi (\beta )}{Z}}\sigma`
    Variance  .. math::
                  \sigma ^{2}\left[1+{\frac {\alpha \phi (\alpha )-\beta \phi (\beta )}{Z}}-
                  \left({\frac {\phi (\alpha )-\phi (\beta )}{Z}}\right)^{2}\right]
    ========  ==========================================

    Parameters
    ----------
    mu : float
        Mean.
    sigma : float
        Standard deviation (sigma > 0)
    lower: float
        Lower limit.
    upper: float
        Upper limit (upper > lower).
    """

    def __init__(self, mu=None, sigma=None, lower=None, upper=None):
        super().__init__()
        self._parametrization(mu, sigma, lower, upper)

    def _parametrization(self, mu=None, sigma=None, lower=None, upper=None):
        self.mu = mu
        self.sigma = sigma
        self.lower = lower
        self.upper = upper
        self.params = (self.mu, self.sigma, self.lower, self.upper)
        self.param_names = ("mu", "sigma", "lower", "upper")
        self.params_support = (
            (-np.inf, np.inf),
            (eps, np.inf),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
        )
        if lower is None:
            self.lower = -np.inf
        if upper is None:
            self.upper = np.inf
        self.support = (self.lower, self.upper)
        if all_not_none(mu, sigma, lower, upper):
            self._update(mu, sigma, lower, upper)

    def _update(self, mu, sigma, lower=None, upper=None):
        if lower is not None:
            self.lower = np.float64(lower)
        if upper is not None:
            self.upper = np.float64(upper)

        self.mu = np.float64(mu)
        self.sigma = np.float64(sigma)
        self.params = (self.mu, self.sigma, self.lower, self.upper)
        self.support = (self.lower, self.upper)
        self.is_frozen = True

    def pdf(self, x):
        """
        Compute the probability density function (PDF) at a given point x.
        """
        x = np.asarray(x)
        return np.exp(nb_logpdf(x, self.mu, self.sigma, self.lower, self.upper))

    def cdf(self, x):
        """
        Compute the cumulative distribution function (CDF) at a given point x.
        """
        x = np.asarray(x)
        return nb_cdf(x, self.mu, self.sigma, self.lower, self.upper)

    def ppf(self, q):
        """
        Compute the percent point function (PPF) at a given probability q.
        """
        q = np.asarray(q)
        return nb_ppf(q, self.mu, self.sigma, self.lower, self.upper)

    def logpdf(self, x):
        """
        Compute the log probability density function (log PDF) at a given point x.
        """
        return nb_logpdf(x, self.mu, self.sigma, self.lower, self.upper)

    def _neg_logpdf(self, x):
        """
        Compute the neg log_pdf sum for the array x.
        """
        return nb_neg_logpdf(x, self.mu, self.sigma, self.lower, self.upper)

    def entropy(self):
        return nb_entropy(self.mu, self.sigma, self.lower, self.upper)

    def mean(self):
        alpha = (self.lower - self.mu) / self.sigma
        beta = (self.upper - self.mu) / self.sigma
        z_val = 0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))
        return (
            self.mu
            + (
                (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * alpha**2))
                - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * beta**2))
            )
            / z_val
            * self.sigma
        )

    def median(self):
        alpha = (self.lower - self.mu) / self.sigma
        beta = (self.upper - self.mu) / self.sigma
        inv_phi = 2**0.5 * erfinv(
            2 * ((0.5 * (1 + erf(alpha / 2**0.5)) + 0.5 * (1 + erf(beta / 2**0.5))) / 2) - 1
        )
        return self.mu + inv_phi * self.sigma

    def var(self):
        alpha = (self.lower - self.mu) / self.sigma
        beta = (self.upper - self.mu) / self.sigma
        z_val = 0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))
        # Handle for -np.inf or np.inf
        psi_alpha = (0, 0) if alpha == -np.inf else (1, alpha)
        psi_beta = (0, 0) if beta == np.inf else (1, beta)
        return self.sigma**2 * (
            1
            - (
                psi_beta[1] * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                - psi_alpha[1] * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
            )
            / z_val
            - (
                (
                    (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[0]
                    - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[0]
                )
                / z_val
            )
            ** 2
        )

    def std(self):
        return self.var() ** 0.5

    def skewness(self):
        alpha = (self.lower - self.mu) / self.sigma
        beta = (self.upper - self.mu) / self.sigma
        z_val = 0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))
        # Handle for -np.inf or np.inf
        psi_alpha = (0, 0) if alpha == -np.inf else (1, alpha)
        psi_beta = (0, 0) if beta == np.inf else (1, beta)
        numerator = (
            (
                (psi_alpha[1] ** 2 - 1)
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
                * psi_alpha[0]
                - (psi_beta[1] ** 2 - 1)
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                * psi_beta[0]
            )
            / z_val
            - 3
            * (
                psi_alpha[1]
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
                * psi_alpha[0]
                - psi_beta[1]
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                * psi_beta[0]
            )
            * (
                (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[0]
                - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[0]
            )
            / z_val**2
            + 2
            * (
                (
                    (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[0]
                    - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[0]
                )
                / z_val
            )
            ** 3
        )
        denominator = (
            1
            + (
                psi_alpha[1]
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
                * psi_alpha[0]
                - psi_beta[1]
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                * psi_beta[0]
            )
            / z_val
            - (
                (
                    (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[0]
                    - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[0]
                )
                / z_val
            )
            ** 2
        ) ** (3 / 2)
        return numerator / denominator

    def kurtosis(self):
        alpha = (self.lower - self.mu) / self.sigma
        beta = (self.upper - self.mu) / self.sigma
        z_val = 0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))
        # Handle for -np.inf or np.inf
        psi_alpha = (0, 0) if alpha == -np.inf else (1, alpha)
        psi_beta = (0, 0) if beta == np.inf else (1, beta)

        numerator = (
            (
                12
                * (
                    psi_alpha[1]
                    * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
                    * psi_alpha[0]
                    - psi_beta[1]
                    * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                    * psi_beta[0]
                )
                * (
                    (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[0]
                    - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[0]
                )
                ** 2
                / z_val**3
            )
            - (
                4
                * (
                    (psi_alpha[1] ** 2 - 1)
                    * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
                    * psi_alpha[0]
                    - (psi_beta[1] ** 2 - 1)
                    * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                    * psi_beta[0]
                )
                * (
                    (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[0]
                    - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[0]
                )
                / z_val**2
            )
            - (
                3
                * (
                    (
                        psi_alpha[1]
                        * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
                        * psi_alpha[0]
                        - psi_beta[1]
                        * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                        * psi_beta[0]
                    )
                    / z_val
                )
                ** 2
            )
            - (
                6
                * (
                    (
                        (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[0]
                        - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[0]
                    )
                    / z_val
                )
                ** 4
            )
            + (
                (psi_alpha[1] ** 3 - 3 * psi_alpha[1])
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
                * psi_alpha[0]
                - (psi_beta[1] ** 3 - 3 * psi_beta[1])
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                * psi_beta[0]
            )
            / z_val
        )

        denominator = (
            1
            + (
                psi_alpha[1]
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2))
                * psi_alpha[0]
                - psi_beta[1]
                * (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2))
                * psi_beta[0]
            )
            / z_val
            - (
                (
                    (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[0]
                    - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[0]
                )
                / z_val
            )
            ** 2
        ) ** 2

        return numerator / denominator

    def rvs(self, size=None, random_state=None):
        random_state = np.random.default_rng(random_state)
        random_samples = random_state.uniform(0, 1, size)
        return nb_rvs(random_samples, self.mu, self.sigma, self.lower, self.upper)

    def _fit_moments(self, mean, sigma):
        # Assume gaussian
        self._update(mean, sigma)

    def _fit_mle(self, sample):
        self._update(None, None, np.min(sample), np.max(sample))
        optimize_ml(self, sample)


@nb.njit(cache=True)
def nb_cdf(x, mu, sigma, lower, upper):
    xi = (x - mu) / sigma
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    z_val = 0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))
    prob = (0.5 * (1 + erf(xi / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))) / z_val
    return cdf_bounds(prob, x, lower, upper)


@nb.njit(cache=True)
def nb_ppf(q, mu, sigma, lower, upper):
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    inv_phi = 2**0.5 * erfinv(
        2
        * (
            0.5 * (1 + erf(alpha / 2**0.5))
            + q * (0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5)))
        )
        - 1
    )
    return ppf_bounds_cont(inv_phi * sigma + mu, q, lower, upper)


@nb.njit(cache=True)
def nb_entropy(mu, sigma, lower, upper):
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    z_val = 0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))
    # Handle for -np.inf or np.inf
    psi_alpha = (0, 0) if alpha == -np.inf else (1, alpha)
    psi_beta = (0, 0) if beta == np.inf else (1, beta)
    return np.log((2 * np.pi * np.e) ** 0.5 * sigma * z_val) + (
        (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_alpha[1] ** 2)) * psi_alpha[1] * psi_alpha[0]
        - (1 / (2 * np.pi) ** 0.5 * np.exp(-0.5 * psi_beta[1] ** 2)) * psi_beta[1] * psi_beta[0]
    ) / (2 * z_val)


@nb.vectorize(nopython=True, cache=True)
def nb_logpdf(x, mu, sigma, lower, upper):
    if x < lower or x > upper:
        return -np.inf
    else:
        xi = (x - mu) / sigma
        alpha = (lower - mu) / sigma
        beta = (upper - mu) / sigma
        z_val = 0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))
        logphi = np.log(1 / (2 * np.pi) ** 0.5) - xi**2 / 2
        return logphi - (np.log(sigma) + np.log(z_val))


@nb.njit(cache=True)
def nb_neg_logpdf(x, mu, sigma, lower, upper):
    return -(nb_logpdf(x, mu, sigma, lower, upper)).sum()


@nb.njit(cache=True)
def nb_rvs(random_samples, mu, sigma, lower, upper):
    alpha = (lower - mu) / sigma
    beta = (upper - mu) / sigma
    z_val = 0.5 * (1 + erf(beta / 2**0.5)) - 0.5 * (1 + erf(alpha / 2**0.5))
    inv_phi = 2**0.5 * erfinv(
        2 * (0.5 * (1 + erf(alpha / 2**0.5)) + random_samples * z_val) - 1
    )
    return inv_phi * sigma + mu
