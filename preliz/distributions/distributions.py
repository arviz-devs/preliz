"""
Parent classes for all families.
"""
# pylint: disable=no-member
from collections import namedtuple

import numpy as np

from ..utils.plot_utils import plot_pdfpmf, plot_cdf, plot_ppf
from ..utils.utils import hdi_from_pdf


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
        """Update the rv_frozen object"""

        frozen = self._get_frozen()
        if frozen is not None:
            self.is_frozen = True
            self.rv_frozen = frozen
            self.rv_frozen.name = self.name

    def summary(self, fmt=".2f", mass=0.94):
        """
        Namedtuple with the mean, median, standard deviation, and lower and upper bounds
        of the equal-tailed interval.

        Parameters
        ----------
        fmt : str
            fmt used to represent results using f-string fmt for floats. Default to ".2f"
            i.e. 2 digits after the decimal point.
        mass: float
            Probability mass for the equal-tailed interval. Defaults to 0.94
        """
        if self.is_frozen:
            attr = namedtuple(self.name.capitalize(), ["mean", "median", "std", "lower", "upper"])
            mean = float(f"{self.rv_frozen.mean():{fmt}}")
            median = float(f"{self.rv_frozen.median():{fmt}}")
            std = float(f"{self.rv_frozen.std():{fmt}}")
            eti = self.rv_frozen.interval(mass)
            lower_tail = float(f"{eti[0]:{fmt}}")
            upper_tail = float(f"{eti[1]:{fmt}}")
            return attr(mean, median, std, lower_tail, upper_tail)
        else:
            return None

    def rvs(self, size=1, random_state=None):
        """Random sample

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates. Defaults to 1.
        random_state : {None, int, numpy.random.Generator, numpy.random.RandomState}
            Defaults to None
        """
        return self.rv_frozen.rvs(size=size, random_state=random_state)

    def eti(self, fmt=".2f", mass=0.94):
        """Equal-tailed interval containing `mass`.

        Parameters
        ----------
        fmt : str
            fmt used to represent results using f-string fmt for floats. Default to ".2f"
            i.e. 2 digits after the decimal point.
        mass: float
            Probability mass in the interval. Defaults to 0.94
        """
        eti = self.rv_frozen.interval(mass)
        lower_tail = float(f"{eti[0]:{fmt}}")
        upper_tail = float(f"{eti[1]:{fmt}}")
        return (lower_tail, upper_tail)

    def hdi(self, fmt=".2f", mass=0.94):
        """Highest density interval containing `mass`.

        Parameters
        ----------
        fmt : str
            fmt used to represent results using f-string fmt for floats. Default to ".2f"
            i.e. 2 digits after the decimal point.
        mass: float
            Probability mass in the interval. Defaults to 0.94
        """
        hdi = hdi_from_pdf(self, mass)
        lower_tail = float(f"{hdi[0]:{fmt}}")
        upper_tail = float(f"{hdi[1]:{fmt}}")
        return (lower_tail, upper_tail)

    def _check_endpoints(self, lower, upper, raise_error=True):
        """
        Evaluate if the lower and upper values are in the support of the distribution

        Parameters
        ----------
        lower : int or float
            lower endpoint
        upper : int or float
            upper endpoint
        raise_error : bool
            If True (default) it will raise ValueErrors, otherwise it will return True
            if the lower and upper endpoint are in the support of the distribution or
            False otherwise.
        """
        s_l, s_u = self.support

        if raise_error:
            domain_error = (
                f"The provided endpoints are outside the domain of the {self.name} distribution"
            )

            if np.isfinite(s_l):
                if lower < s_l:
                    raise ValueError(domain_error)

            if np.isfinite(s_u):
                if upper > s_u:
                    raise ValueError(domain_error)
            if np.isfinite(s_l) and np.isfinite(s_u):
                if lower == s_l and upper == s_u:
                    raise ValueError(
                        "Given the provided endpoints, mass will be always 1. "
                        "Please provide other values"
                    )
            return None
        else:
            return lower >= s_l and upper <= s_u

    def _finite_endpoints(self, support):
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
            lower_ep, upper_ep = self.support

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
            for the mean (μ), standard deviation (σ), variance (σ²), skew (γ) or kurtosis (κ)
            respectively. Other strings will be ignored. Defaults to None.
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        quantiles : list
            Values of the five quantiles to use when ``pointinterval=True`` if None (default)
            the values ``[0.05, 0.25, 0.5, 0.75, 0.95]`` will be used. The number of elements
            should be 5, 3, 1 or 0 (in this last case nothing will be plotted).
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

    def plot_cdf(
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
        Plot the cumulative distribution function.

        Parameters
        ----------
        moments : str
            Compute moments. Use any combination of the strings ``m``, ``d``, ``v``, ``s`` or ``k``
            for the mean (μ), standard deviation (σ), variance (σ²), skew (γ) or kurtosis (κ)
            respectively. Other strings will be ignored. Defaults to None.
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        quantiles : list
            Values of the five quantiles to use when ``pointinterval=True`` if None (default)
            the values ``[0.05, 0.25, 0.5, 0.75, 0.95]`` will be used. The number of elements
            should be 5, 3, 1 or 0 (in this last case nothing will be plotted).
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
            return plot_cdf(self, moments, pointinterval, quantiles, support, legend, figsize, ax)
        else:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )

    def plot_ppf(
        self,
        moments=None,
        pointinterval=False,
        quantiles=None,
        legend="legend",
        figsize=None,
        ax=None,
    ):
        """
        Plot the quantile function.

        Parameters
        ----------
        moments : str
            Compute moments. Use any combination of the strings ``m``, ``d``, ``v``, ``s`` or ``k``
            for the mean (μ), standard deviation (σ), variance (σ²), skew (γ) or kurtosis (κ)
            respectively. Other strings will be ignored. Defaults to None.
        pointinterval : bool
            Whether to include a plot of the quantiles. Defaults to False. If True the default is to
            plot the median and two interquantiles ranges.
        quantiles : list
            Values of the five quantiles to use when ``pointinterval=True`` if None (default)
            the values ``[0.05, 0.25, 0.5, 0.75, 0.95]`` will be used. The number of elements
            should be 5, 3, 1 or 0 (in this last case nothing will be plotted).
        legend : str
            Whether to include a string with the distribution and its parameter as a ``"legend"`` a
            ``"title"`` or not include them ``None``.
        figsize : tuple
            Size of the figure
        ax : matplotlib axes
        """
        if self.is_frozen:
            return plot_ppf(self, moments, pointinterval, quantiles, legend, figsize, ax)
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

    def xvals(self, support):
        """Provide x values in the support of the distribution. This is useful for example when
        plotting.

        Parameters
        ----------
        support : str
            Available options are "full" or "restricted".
        """
        lower_ep, upper_ep = self._finite_endpoints(support)
        x_vals = np.linspace(lower_ep, upper_ep, 1000)
        return x_vals

    def _fit_mle(self, sample, **kwargs):
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
        lower_ep, upper_ep = self._finite_endpoints(support)
        x_vals = np.arange(lower_ep, upper_ep + 1, dtype=int)
        return x_vals
