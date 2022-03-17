import numpy as np
from scipy import stats
from scipy.optimize import least_squares


dist_dict = {
    "beta": ("alpha", "beta"),
    "exponential": ("lam",),
    "gamma": ("alpha", "beta"),
    "lognormal": ("mu", "sigma"),
    "normal": ("mu", "sigma"),
    "student": ("nu", "mu", "sigma"),
}  # These names could be different for different "parametrizations"


def get_parametrization(name, a, b, extra, parametrization):
    """Generates a string with the name of a distribution and its parameters

    Parameters
    ----------
    name : str
        Name of a distribution.
    a : float
        First parameter under optimization
    b : float
        Second parameter under optimization
    extra : float
        Extra parameter that is kept fixed. For example the degrees of freedom of the Student t
    parametrization : str
        Which parametrization to use, available options are PyMC and Scipy
    """
    if parametrization == "pymc":
        if name == "gamma":
            title = f"{name}({dist_dict[name][0]}={a:.2f}, {dist_dict[name][1]}={1/b:.2f})"
        elif name == "exponential":
            title = f"{name}({dist_dict[name][0]}={1/a:.2f})"
        elif name == "lognormal":
            title = f"{name}({dist_dict[name][0]}={a:.2f}, {dist_dict[name][1]}={b:.2f})"
        elif name in ["normal", "beta"]:
            title = f"{name}({dist_dict[name][0]}={a:.2f}, {dist_dict[name][1]}={b:.2f})"
        elif name == "student":
            title = (
                f"{name}({dist_dict[name][0]}={extra:.2f}, {dist_dict[name][1]}={a:.2f}, "
                f"{dist_dict[name][2]}={b:.2f})"
            )
    elif parametrization == "scipy":
        if name in ["gamma", "lognormal", "normal"]:
            title = f"{name}({dist_dict[name][0]}={a:.2f}, {dist_dict[name][1]}={b:.2f})"
        elif name == "exponential":
            title = f"{name}({dist_dict[name][0]}={a:.2f})"
        elif name == "student":
            title = (
                f"{name}({dist_dict[name][0]}={extra:.2f}, {dist_dict[name][1]}={a:.2f}, "
                f"{dist_dict[name][2]}={b:.2f})"
            )
    return title


def check_boundaries(name, lower, upper):
    """

    Parameters
    ----------
    name : str
        Name of a distribution.
    lower : float
        Lower bound.
    upper: float
        Upper bound.
    """
    domain_error = f"The provided boundaries are outside the domain of the {name} distribution"
    if name == "beta":
        if lower == 0 and upper == 1:
            raise ValueError(
                "Given the provided boundaries, mass will be always 1. Please provide other values"
            )
        if lower < 0 or upper > 1:
            raise ValueError(domain_error)
    elif name in ["exponential", "gamma", "lognormal"]:
        if lower < 0:
            raise ValueError(domain_error)


def relative_error(rv_frozen, upper, lower, requiered_mass):
    computed_mass = rv_frozen.cdf(upper) - rv_frozen.cdf(lower)
    return (computed_mass - requiered_mass) / requiered_mass * 100


def sane_scipy(dist, a, b, extra=None):
    dist_name = dist.name
    if dist_name in ["norm", "beta"]:
        rv_frozen = dist(a, b)
    elif dist_name == "gamma":
        rv_frozen = dist(a=a, scale=b)
    elif dist_name == "lognorm":
        rv_frozen = dist(b, scale=np.exp(a))
    elif dist_name == "expon":
        rv_frozen = dist(scale=a)
    elif dist_name == "t":
        rv_frozen = dist(df=extra, loc=a, scale=b)

    rv_frozen.name = dist_name
    return rv_frozen


def compute_xvals(rv_frozen):
    if np.isfinite(rv_frozen.a):
        lq = rv_frozen.a
    else:
        lq = 0.001

    if np.isfinite(rv_frozen.b):
        uq = rv_frozen.b
    else:
        uq = 0.999

    x = np.linspace(rv_frozen.ppf(lq), rv_frozen.ppf(uq), 1000)
    return x


def cdf_loss(params, dist, lower, upper, mass, extra=None):
    """
    Cumulative distribution Loss function

    Parameters
    ----------
    params : tuple
        Parameters under optimization.
    dist : scipy distribution
        Distribution to optimize.
    lower : float
        Lower bound.
    upper: float
        Upper bound.
    mass: float
        Probability mass between ``lower`` and ``upper`` bounds.
    extra : float
        Extra parameter that is kept fixed. For example the degrees of freedom of the Student t.

    Returns
    -------
    cdf_loss
        difference between the cdf (between lower and upper bound) and the requiered mass.
    """
    a, b = params
    rv_frozen = sane_scipy(dist, a, b, extra)
    cdf0 = rv_frozen.cdf(lower)
    cdf1 = rv_frozen.cdf(upper)
    cdf_loss = (cdf1 - cdf0) - mass
    return cdf_loss


def optimize(lower, upper, mass, dist=None, a=None, b=None, extra=None):
    """Use least squares to perform a constraints optimization

    Parameters
    ----------
    lower : float
        Lower bound.
    upper: float
        Upper bound.
    mass: float
        Probability mass between ``lower`` and ``upper`` bounds.
    dist : scipy distribution
        Distribution to optimize.
    a : float
        First parameter under optimization.
    b : float
        Second parameter under optimization.
    extra : float
        Extra parameter that is kept fixed. For example the degrees of freedom of the Student t.

    Returns
    -------
    opt : scipy.optimize.OptimizeResult
    """
    if dist is None:
        mu_init = (lower + upper) / 2
        sigma_init = ((upper - lower) / 4) / mass
        opt = least_squares(
            cdf_loss,
            x0=(mu_init, sigma_init),
            method="dogbox",
            args=(stats.norm, lower, upper, mass),
        )
    else:
        opt = least_squares(
            cdf_loss, x0=(a, b), args=(dist, lower, upper, mass, extra), method="dogbox"
        )
    return opt


def method_of_moments(name, mu, sigma):
    """Use mean and standard deviation values to estimate parameters of a given distribution

    Parameters
    ----------
    name : str
        Name of a distribution.
    mu : float
        mean value.
    sigma : float
        standard deviation.

    Returns
    -------
    a : float
        First parameter.
    b : float
        Second parameter.
    dist : scipy distribution.
    """
    if name == "beta":
        kappa = (mu * (1 - mu) / (sigma) ** 2) - 1
        a = max(0.5, mu * kappa)
        b = max(0.5, (1 - mu) * kappa)
        dist = stats.beta

    elif name == "lognormal":
        a = np.log(mu**2 / (sigma**2 + mu**2) ** 0.5)
        b = np.log(sigma**2 / mu**2 + 1) ** 0.5
        dist = stats.lognorm

    elif name == "exponential":
        a = mu
        b = sigma
        dist = stats.expon

    elif name == "gamma":
        a = mu**2 / sigma**2
        b = sigma**2 / mu
        dist = stats.gamma
    elif name == "student":
        a = mu
        b = sigma
        dist = stats.t
    else:
        raise NotImplementedError(f"The distribution {name} is not implemented")
    return a, b, dist
