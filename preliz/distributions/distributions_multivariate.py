"""
Parent classes for multivariate families.
"""
# pylint: disable=no-member
from collections import namedtuple

import numpy as np
from scipy.special import betainc  # pylint: disable=no-name-in-module

from ..internal.distribution_helper import (
    valid_scalar_params,
    valid_distribution,
)


class Multivariate:
    """
    Base class for Multivariate distributions.

    Not intended for direct instantiation.
    """

    def __init__(self):
        self.rv_frozen = None
        self.is_frozen = False
        self.opt = None

    def __repr__(self):
        name = self.__class__.__name__
        if self.is_frozen:
            bolded_name = "\033[1m" + name + "\033[0m"

            description = "".join(
                f"{n}={np.round(v, 2)}," for n, v in zip(self.param_names, self.params)
            ).strip(",")

            return f"{bolded_name}({description})"
        else:
            return name

    def _update_rv_frozen(self):
        """Update the rv_frozen object"""

        frozen = self._get_frozen()
        if frozen is not None:
            self.is_frozen = True
            self.rv_frozen = frozen

    def summary(self):
        """
        Namedtuple with the mean, and standard deviation of the distribution.
        """
        valid_distribution(self)

        if valid_scalar_params(self):
            attr = namedtuple(self.__class__.__name__, ["mean", "std"])
            if self.__class__.__name__ == "MvNormal":
                # for some weird reason, the mean of mvnormal is an attribute
                # and not a method like for the rest of the distributions
                mean = self.rv_frozen.mean
            else:
                mean = self.rv_frozen.mean()
            std = self.rv_frozen.var() ** 0.5
            return attr(mean, std)
        else:
            return None

    def rvs(self, *args, **kwds):
        """Random sample

        Parameters
        ----------
        size : int or tuple of ints, optional
            Defining number of random variates. Defaults to 1.
        random_state : {None, int, numpy.random.Generator, numpy.random.RandomState}
            Defaults to None
        """
        return self.rv_frozen.rvs(*args, **kwds)

    def cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the cdf
        """
        # not sure if this is right
        return betainc(self.alpha, self.alpha.sum(), x)

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
                f"The provided endpoints are outside the domain of the "
                f"{self.__class__.__name__} distribution"
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
                lower_ep = self.ppf(0.0001)
            if not np.isfinite(upper_ep) or support == "restricted":
                upper_ep = self.ppf(0.9999)

        return lower_ep, upper_ep


class Continuous(Multivariate):
    """Base class for continuous multivariate distributions."""

    def __init__(self):
        super().__init__()
        self.kind = "continuous_multivariate"

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

    def pdf(self, x):
        """Probability density function at x.

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the pdf
        """
        return self.rv_frozen.pdf(x)


class Discrete(Multivariate):
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

    def pdf(self, x, *args, **kwds):
        """Probability mass function at x.

        Parameters
        ----------
        x : array_like
            Values on which to evaluate the pdf
        """
        return self.rv_frozen.pmf(x, *args, **kwds)
