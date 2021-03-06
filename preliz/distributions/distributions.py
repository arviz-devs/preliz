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

    def _update_rv_frozen(self):
        """Update the rv_frozen object when the parameters are not None"""
        self.is_frozen = any(self.params)
        if self.is_frozen:
            self.rv_frozen = self._get_frozen()
            self.rv_frozen.name = self.name

    def check_endpoints(self, lower, upper):
        """
        Evaluate if the lower and upper values are in the support of the distribution

        Parameters
        ----------
        support : str
            Available options are "full" or "restricted".
        lower : int or float
            lower endpoint
        upper : int or float
            upper endpoint
        """
        domain_error = (
            f"The provided endpoints are outside the domain of the {self.name} distribution"
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
                    "Given the provided endpoints, mass will be always 1. "
                    "Please provide other values"
                )

    def finite_endpoints(self, support):
        """
        Return finite endpoints even for unbounded distributions

        Parameters
        ----------
        support : str or tuple
            Available string options are "full" or "restricted".
        """
        if isinstance(support, tuple):
            lower_ep, upper_ep = support
        else:
            lower_ep = self.rv_frozen.a
            upper_ep = self.rv_frozen.b

            if not np.isfinite(lower_ep) or support == "restricted":
                lower_ep = self.rv_frozen.ppf(0.0001)
            if not np.isfinite(upper_ep) or support == "restricted":
                upper_ep = self.rv_frozen.ppf(0.9999)

        return lower_ep, upper_ep

    def plot_pdf(
        self,
        moments=None,
        pointinterval=False,
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
        moments : str
            Compute moments. Use any combination of the strings ``m``, ``d``, ``v``, ``s`` or ``k``
            for the mean (??), standard deviation (??), variance (????), skew (??) or kurtosis (??)
            respectively. Other strings will be ignored. Defaults to None.
        pointinterval : bool
            Whether to include a plot of the mean as a dot and two inter-quantile ranges as
            lines. Defaults to False.
        quantiles : list
            Values of the four quantiles to use when ``pointinterval=True`` if None (default)
            the values ``[0.05, 0.25, 0.75, 0.95]`` will be used.
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
            return plot_pdfpmf(
                self, moments, pointinterval, quantiles, support, legend, figsize, ax
            )
        else:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )

    def plot_cdf(self, moments=None, support="full", legend="legend", figsize=None, ax=None):
        """
        Plot the cumulative distribution function.

        Parameters
        ----------
        moments : str
            Compute moments. Use any combination of the strings ``m``, ``d``, ``v``, ``s`` or ``k``
            for the mean (??), standard deviation (??), variance (????), skew (??) or kurtosis (??)
            respectively. Other strings will be ignored. Defaults to None.
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
            return plot_cdf(self, moments, support, legend, figsize, ax)
        else:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )

    def plot_ppf(self, moments=None, legend="legend", figsize=None, ax=None):
        """
        Plot the quantile function.

        Parameters
        ----------
        moments : str
            Compute moments. Use any combination of the strings ``m``, ``d``, ``v``, ``s`` or ``k``
            for the mean (??), standard deviation (??), variance (????), skew (??) or kurtosis (??)
            respectively. Other strings will be ignored. Defaults to None.
        legend : str
            Whether to include a string with the distribution and its parameter as a ``"legend"`` a
            ``"title"`` or not include them ``None``.
        figsize : tuple
            Size of the figure
        ax : matplotlib axes
        """
        if self.is_frozen:
            return plot_ppf(self, moments, legend, figsize, ax)
        else:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )

    def _fit_moments(self, mean, sigma):
        """
        Estimate the parameters of the distribution from the mean and standard deviation.

        Parameters
        ----------
        mean : float
            mean value
        sigma : float
            standard deviation
        """
        raise NotImplementedError


class Continuous(Distribution):
    """Base class for continuous distributions."""

    def __init__(self):
        super().__init__()
        self.kind = "continuous"

    def xvals(self, support):
        """Provide x values in the support of the distribution. This is useful for example when
        plotting.

        Parameters
        ----------
        support : str
            Available options are "full" or "restricted".
        """
        lower_ep, upper_ep = self.finite_endpoints(support)
        x_vals = np.linspace(lower_ep, upper_ep, 1000)
        return x_vals

    def fit_mle(self, sample, **kwargs):
        """
        Estimate the parameters of the distribution from a sample by maximizing the likelihood.

        Parameters
        ----------
        sample : array-like
            a sample
        kwargs : dict
            keywords arguments passed to scipy.stats.rv_continuous.fit
        """
        raise NotImplementedError


class Discrete(Distribution):
    """Base class for discrete distributions."""

    def __init__(self):
        super().__init__()
        self.kind = "discrete"

    def xvals(self, support):
        """Provide x values in the support of the distribution. This is useful for example when
        plotting.
        """
        lower_ep, upper_ep = self.finite_endpoints(support)
        x_vals = np.arange(lower_ep, upper_ep + 1, dtype=int)
        return x_vals

    def fit_mle(self, sample):
        """
        Estimate the parameters of the distribution from a sample by maximizing the likelihood.

        sample : array-like
            a sample
        """
        raise NotImplementedError
