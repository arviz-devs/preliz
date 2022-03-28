"""
Parent classes for all families.
"""
# pylint: disable=no-member
import numpy as np
from ..utils.plot_utils import plot_dist


class Distribution:
    """
    Base class for distributions.

    Not intended for direct instantiation.
    """

    def __init__(self):
        self.rv_frozen = None
        self.is_frozen = False
        self.opt = None

    def _cdf_loss(self, params, lower, upper, mass):
        self._update(*params)
        rv_frozen = self.rv_frozen
        cdf0 = rv_frozen.cdf(lower)
        cdf1 = rv_frozen.cdf(upper)
        loss = (cdf1 - cdf0) - mass
        return loss

    def _check_boundaries(self, lower, upper):
        domain_error = (
            f"The provided boundaries are outside the domain of the {self.name} distribution"
        )
        if np.isfinite(self.dist.a):
            if lower < self.dist.a:
                raise ValueError(domain_error)

        if np.isfinite(self.dist.b):
            if upper > self.dist.b:
                raise ValueError(domain_error)
        if np.isfinite(self.dist.a) and np.isfinite(self.dist.b):
            if lower == self.dist.a and upper == self.dist.b:
                raise ValueError(
                    "Given the provided boundaries, mass will be always 1. "
                    "Please provide other values"
                )

    def _update_rv_frozen(self):
        self.is_frozen = any(self.params)
        if self.is_frozen:
            self.rv_frozen = self._get_frozen()
            self.rv_frozen.name = self.name

    def _xvals(self):
        if np.isfinite(self.rv_frozen.a):
            lq = self.rv_frozen.a
        else:
            lq = 0.001

        if np.isfinite(self.rv_frozen.b):
            uq = self.rv_frozen.b
        else:
            uq = 0.999

        if self.kind == "continuous":
            x = np.linspace(self.rv_frozen.ppf(lq), self.rv_frozen.ppf(uq), 1000)
        else:
            x = np.arange(int(self.rv_frozen.ppf(lq)), int(self.rv_frozen.ppf(uq)))
        return x

    def plot(self, box=False, quantiles=None, figsize=None, ax=None):
        if self.is_frozen:
            return plot_dist(self, box, quantiles, figsize, ax)
        else:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )


class Continuous(Distribution):
    def __init__(self):
        super().__init__()
        self.kind = "continuous"


class Discrete(Distribution):
    def __init__(self):
        super().__init__()
        self.kind = "discrete"
