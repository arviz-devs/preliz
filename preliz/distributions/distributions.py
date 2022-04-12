"""
Parent classes for all families.
"""
# pylint: disable=no-member
import numpy as np
from ..utils.plot_utils import plot_pdfpmf, plot_cdf, plot_ppf


class Distribution:
    """
    Base class for distributions.

    Not intended for direct instantiation.
    """

    def __init__(self):
        self.rv_frozen = None
        self.is_frozen = False
        self.opt = None

    def __repr__(self):
        if self.is_frozen:
            bolded_name = "\033[1m" + self.name.capitalize() + "\033[0m"
            description = "".join(
                f"{n}={v:.2f}," for n, v in zip(self.param_names, self.params)
            ).strip(",")
            return f"{bolded_name}({description})"
        else:
            return self.name

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

    def _finite_endpoints(self, support):
        """
        Return finite end-points even for unbounded distributions
        """
        lower_ep = self.rv_frozen.a
        upper_ep = self.rv_frozen.b

        if not np.isfinite(lower_ep) or support == "restricted":
            lower_ep = self.rv_frozen.ppf(0.0001)
        if not np.isfinite(upper_ep) or support == "restricted":
            upper_ep = self.rv_frozen.ppf(0.9999)

        return lower_ep, upper_ep

    def plot_pdf(
        self,
        box=False,
        quantiles=None,
        support="full",
        legend="legend",
        figsize=None,
        ax=None,
    ):
        """
        Plot the  pdf (continuous) or pmf (discrete).

        Parameters
        ----------
        box : bool
            Whether to include a plot of the mean as a dot and two inter-quantile ranges as
            lines. Defaults to False.
        quantiles : list
            Values of the four quantiles to use when ``box=True`` if None (default) the values
            will be used ``[0.05, 0.25, 0.75, 0.95]``.
        support : str:
            If ``full`` use the finite end-points to set the limits of the plot. For unbounded
            end-points or if ``restricted`` use the 0.001 and 0.999 quantiles to set the limits.
        legend : str
            Whether to include a string with the distribution and its parameter as a ``"legend"`` a
            ``"title"`` or not include them ``None``.
        figsize : tuple
            Size of the figure
        ax : matplotlib axes
        """
        if self.is_frozen:
            return plot_pdfpmf(self, box, quantiles, support, legend, figsize, ax)
        else:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )

    def plot_cdf(self, support="full", legend="legend", figsize=None, ax=None):
        """
        Plot the cumulative distribution function.

        Parameters
        ----------
        support : str:
            If ``full`` use the finite end-points to set the limits of the plot. For unbounded
            end-points or if ``restricted`` use the 0.001 and 0.999 quantiles to set the limits.
        legend : str
            Whether to include a string with the distribution and its parameter as a ``"legend"`` a
            ``"title"`` or not include them ``None``.
        figsize : tuple
            Size of the figure
        ax : matplotlib axes
        """
        if self.is_frozen:
            return plot_cdf(self, support, legend, figsize, ax)
        else:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )

    def plot_ppf(self, legend="legend", figsize=None, ax=None):
        """
        Plot the quantile function.

        Parameters
        ----------
        legend : str
            Whether to include a string with the distribution and its parameter as a ``"legend"`` a
            ``"title"`` or not include them ``None``.
        figsize : tuple
            Size of the figure
        ax : matplotlib axes
        """
        if self.is_frozen:
            return plot_ppf(self, legend, figsize, ax)
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

    def _xvals(self, support):
        """Provide x values in the support of the distribution. This is useful for example when
        plotting.
        """
        lower_ep, upper_ep = self._finite_endpoints(support)
        x_vals = np.linspace(lower_ep, upper_ep, 1000)
        return x_vals


class Discrete(Distribution):
    """Base class for discrete distributions."""

    def __init__(self):
        super().__init__()
        self.kind = "discrete"

    def _xvals(self, support):
        """Provide x values in the support of the distribution. This is useful for example when
        plotting.
        """
        lower_ep, upper_ep = self._finite_endpoints(support)
        x_vals = np.arange(lower_ep, upper_ep + 1, dtype=int)
        return x_vals
