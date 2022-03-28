"""
Continuous probability distributions.
"""
import numpy as np
from scipy import stats

from .distributions import Continuous
from ..utils.constraints_utils import optimize


class Beta(Continuous):
    def __init__(self, alpha=None, beta=None):

        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.name = "beta"
        self.params = (self.alpha, self.beta)
        self.dist = stats.beta
        self._update_rv_frozen()

    def __repr__(self):
        name = self.name
        if self.is_frozen:
            return f"{name}(alpha={self.alpha:.2f}, beta={self.beta:.2f})"
        else:
            return name

    def _get_frozen(self):
        return self.dist(self.alpha, self.beta)

    def _optimize(self, lower, upper, mass):
        self.opt = optimize(self, self.params, lower, upper, mass)
        alpha, beta = self.opt["x"]
        self._update(alpha, beta)

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        kappa = (mean * (1 - mean) / (sigma) ** 2) - 1
        alpha = max(0.5, mean * kappa)
        beta = max(0.5, (1 - mean) * kappa)
        self._update(alpha, beta)

    def fit_mle(self, sample, **kwargs):
        alpha, beta, _, _ = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)


class Exponential(Continuous):
    def __init__(self, lam=None):
        super().__init__()
        self.lam = lam
        self.name = "exponential"
        self.params = (self.lam, None)
        self.dist = stats.expon
        self._update_rv_frozen()

    def __repr__(self):
        name = self.name
        if self.is_frozen:
            return f"{name}(mu={1/self.lam:.2f})"
        else:
            return name

    def _get_frozen(self):
        return self.dist(scale=self.lam)

    def _optimize(self, lower, upper, mass):
        self.opt = optimize(self, self.params[0], lower, upper, mass)
        lam = self.opt["x"][0]
        self._update(lam)

    def _update(self, lam):
        self.lam = lam
        self.params = (self.lam, None)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma=None):  # pylint: disable=unused-argument
        lam = mean
        self._update(lam)

    def fit_mle(self, sample, **kwargs):
        lam, _ = self.dist.fit(sample, **kwargs)
        self._update(lam)


class Gamma(Continuous):
    def __init__(self, alpha=None, beta=None):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.name = "gamma"
        self.params = (self.alpha, self.beta)
        self.dist = stats.gamma
        self._update_rv_frozen()

    def __repr__(self):
        name = self.name
        if self.is_frozen:
            return f"{name}(alpha={self.alpha:.2f}, beta={1/self.beta:.2f})"
        else:
            return name

    def _get_frozen(self):
        return self.dist(a=self.alpha, scale=self.beta)

    def _optimize(self, lower, upper, mass):
        self.opt = optimize(self, self.params, lower, upper, mass)
        alpha, beta = self.opt["x"]
        self._update(alpha, beta)

    def _update(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.params = (self.alpha, self.beta)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        alpha = mean**2 / sigma**2
        beta = sigma**2 / mean
        self._update(alpha, beta)

    def fit_mle(self, sample, **kwargs):
        alpha, _, beta = self.dist.fit(sample, **kwargs)
        self._update(alpha, beta)


class LogNormal(Continuous):
    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.name = "lognormal"
        self.params = (self.mu, self.sigma)
        self.dist = stats.lognorm
        self._update_rv_frozen()

    def __repr__(self):
        name = self.name
        if self.is_frozen:
            return f"{name}(mu={self.mu:.2f}, sigma={self.sigma:.2f})"
        else:
            return name

    def _get_frozen(self):
        return self.dist(self.sigma, scale=np.exp(self.mu))

    def _optimize(self, lower, upper, mass):
        self.opt = optimize(self, self.params, lower, upper, mass)
        mu, sigma = self.opt["x"]
        self._update(mu, sigma)

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def fit_moments(self, mean, sigma):
        mu = np.log(mean**2 / (sigma**2 + mean**2) ** 0.5)
        sigma = np.log(sigma**2 / mean**2 + 1) ** 0.5
        self._update(mu, sigma)

    def fit_mle(self, sample, **kwargs):
        (
            sigma,
            _,
            mu,
        ) = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma)


class Normal(Continuous):
    def __init__(self, mu=None, sigma=None):
        super().__init__()
        self.mu = mu
        self.sigma = sigma
        self.name = "normal"
        self.params = (self.mu, self.sigma)
        self.dist = stats.norm
        self._update_rv_frozen()

    def __repr__(self):
        name = self.name
        if self.is_frozen:
            return f"{name}(mu={self.mu:.2f}, sigma={self.sigma:.2f})"
        else:
            return name

    def _get_frozen(self):
        return self.dist(self.mu, self.sigma)

    def _update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.params = (self.mu, self.sigma)
        self._update_rv_frozen()

    def _optimize(self, lower, upper, mass):
        self.opt = optimize(self, self.params, lower, upper, mass)
        mu, sigma = self.opt["x"]
        self._update(mu, sigma)

    def fit_moments(self, mean, sigma):
        self._update(mean, sigma)

    def fit_mle(self, sample, **kwargs):
        mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma)


class Student(Continuous):
    def __init__(self, nu=None, mu=None, sigma=None):
        super().__init__()
        self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.name = "student"
        self.params = (self.nu, self.mu, self.sigma)
        self.dist = stats.t
        self._update_rv_frozen()

    def __repr__(self):
        name = self.name
        if self.is_frozen:
            return f"{name}(nu={self.nu:.2f}, mu={self.mu:.2f}, sigma={self.sigma:.2f})"
        else:
            return name

    def _get_frozen(self):
        return self.dist(self.nu, self.mu, self.sigma)

    def _update(self, mu, sigma, nu=None):
        if nu is not None:
            self.nu = nu
        self.mu = mu
        self.sigma = sigma
        self.params = (self.nu, self.mu, self.sigma)
        self._update_rv_frozen()

    def _optimize(self, lower, upper, mass):
        self.opt = optimize(self, self.params[1:], lower, upper, mass)
        mu, sigma = self.opt["x"]
        self._update(mu, sigma)

    def fit_moments(self, mean, sigma):
        # This is a placeholder!!!
        self._update(mean, sigma)

    def fit_mle(self, sample, **kwargs):
        nu, mu, sigma = self.dist.fit(sample, **kwargs)
        self._update(mu, sigma, nu)
