"""
Discrete probability distributions.
"""
from scipy import stats

from .distributions import Discrete


class Poisson(Discrete):
    def __init__(self, mu=None):
        super().__init__()
        self.mu = mu
        self.name = "Poisson"
        self.params = self.mu
        self.dist = stats.poisson
        self._update_rv_frozen()

    def __repr__(self):
        name = self.name
        if self.is_frozen:
            return f"{name}(mu={self.mu:.2f})"
        else:
            return name

    def _get_frozen(self):
        return self.dist(self.mu)

    def _update(self, mu):
        self.mu = mu
        self.params = self.mu
        self._update_rv_frozen()

    def fit_moments(self, mu):
        self._update(mu)

    def fit_mle(self, sample, **kwargs):
        # This is not available from scipy. We will use our own implementation
        raise NotImplementedError
