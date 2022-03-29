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
        """
        Difference between the cumulative distribuion function in the lower-upper interval with
        respect to a given mass.
        """
        self._update(*params)
        rv_frozen = self.rv_frozen
        cdf0 = rv_frozen.cdf(lower)
        cdf1 = rv_frozen.cdf(upper)
        loss = (cdf1 - cdf0) - mass  # - rv_frozen.entropy()/100
        return loss

    def _check_boundaries(self, lower, upper):
        """Evaluate if the lower and upper values are in the support of the distribution"""
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
        """Update the rv_frozen object when the parameters are not None"""
        self.is_frozen = any(self.params)
        if self.is_frozen:
            self.rv_frozen = self._get_frozen()
            self.rv_frozen.name = self.name

    def _finite_endpoints(self):
        """
        Return finite end-points even for unbounded distributions
        """
        lower_ep = self.rv_frozen.a
        upper_ep = self.rv_frozen.b

        if not np.isfinite(lower_ep):
            lower_ep = self.rv_frozen.ppf(0.001)
        if not np.isfinite(upper_ep):
            upper_ep = self.rv_frozen.ppf(0.999)

        return lower_ep, upper_ep

    def plot(self, box=False, quantiles=None, figsize=None, ax=None):
        """
        Plot the  pdf/pmf.

        Parameters
        ----------
            box : bool
                Whether to incluide a plot of the mean as a dot and two interquantile ranges as
                lines. Defaults to False.
            quantiles : list
                Values of the four quantiles to use when ``box=True`` if None (default) the values
                will be used ``[0.05, 0.25, 0.75, 0.95]``.
            figsize : tuple
                Size of the figure
            ax : matplotlib axes
        """
        if self.is_frozen:
            return plot_dist(self, box, quantiles, figsize, ax)
        else:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )


class Continuous(Distribution):
    """Base class for continuous distributions."""

    def __init__(self):
        super().__init__()
        self.kind = "continuous"

    def _xvals(self):
        """Provide x values in the support of the distribuion. This is useful for example when
        plotting.
        """
        lower_ep, upper_ep = self._finite_endpoints()
        x_vals = np.linspace(lower_ep, upper_ep, 1000)
        return x_vals


class Discrete(Distribution):
    """Base class for discrete distributions."""

    def __init__(self):
        super().__init__()
        self.kind = "discrete"

    def _xvals(self):
        """Provide x values in the support of the distribuion. This is useful for example when
        plotting.
        """
        lower_ep, upper_ep = self._finite_endpoints()
        x_vals = np.arange(lower_ep, upper_ep + 1, dtype=int)
        return x_vals
