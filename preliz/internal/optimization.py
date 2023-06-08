"""
Optimization routines and utilities
"""
from sys import modules
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import i0, i1  # pylint: disable=no-name-in-module
from .distribution_helper import init_vals as default_vals


def optimize_max_ent(dist, lower, upper, mass, none_idx, fixed):
    def prob_bound(params, dist, lower, upper, mass):
        params = get_params(dist, params, none_idx, fixed)
        dist._parametrization(**params)
        if dist.kind == "discrete":
            lower -= 1
        cdf0 = dist.cdf(lower)
        cdf1 = dist.cdf(upper)
        loss = (cdf1 - cdf0) - mass
        return loss

    def entropy_loss(params, dist):
        params = get_params(dist, params, none_idx, fixed)
        dist._parametrization(**params)
        return -dist.rv_frozen.entropy()

    cons = {
        "type": "eq",
        "fun": prob_bound,
        "args": (dist, lower, upper, mass),
    }
    init_vals = np.array(dist.params)[none_idx]
    bounds = np.array(dist.params_support)[none_idx]

    opt = minimize(entropy_loss, x0=init_vals, bounds=bounds, args=(dist), constraints=cons)
    params = get_params(dist, opt["x"], none_idx, fixed)
    dist._parametrization(**params)

    return opt


def get_params(dist, params, none_idx, fixed):
    params_ = {}
    pdx = 0
    fdx = 0
    for idx in range(len(dist.params)):
        name = dist.param_names[idx]
        if idx in none_idx:
            params_[name] = params[pdx]
            pdx += 1
        else:
            params_[name] = fixed[fdx]
            fdx += 1

    return params_


def optimize_quartile(dist, x_vals, none_idx, fixed):
    def func(params, dist, x_vals):
        params = get_params(dist, params, none_idx, fixed)
        dist._parametrization(**params)
        loss = dist.cdf(x_vals) - [0.25, 0.5, 0.75]
        return loss

    init_vals = np.array(dist.params)[none_idx]
    bounds = np.array(dist.params_support)[none_idx]
    bounds = list(zip(*bounds))
    opt = least_squares(func, x0=init_vals, args=(dist, x_vals), bounds=bounds)
    params = get_params(dist, opt["x"], none_idx, fixed)
    dist._parametrization(**params)
    return opt


def optimize_cdf(dist, x_vals, ecdf, none_idx, fixed):
    def func(params, dist, x_vals, ecdf):
        params = get_params(dist, params, none_idx, fixed)
        dist._parametrization(**params)
        loss = dist.cdf(x_vals) - ecdf
        return loss

    init_vals = np.array(dist.params)[none_idx]

    opt = least_squares(func, x0=init_vals, args=(dist, x_vals, ecdf))
    dist._update(*opt["x"])
    loss = opt["cost"]
    return loss


def optimize_moments(dist, mean, sigma, params=None):
    def func(params, dist, mean, sigma):
        params = get_params(dist, params, none_idx, fixed)
        dist._parametrization(**params)
        loss = abs(dist.rv_frozen.mean() - mean) + abs(dist.rv_frozen.std() - sigma)
        return loss

    none_idx, fixed = get_fixed_params(dist)

    if params is not None:
        dist._update(*params)
    else:
        dist._update(**default_vals[dist.__class__.__name__])

    init_vals = np.array(dist.params)[none_idx]

    if dist.__class__.__name__ in ["HyperGeometric", "BetaBinomial"]:
        opt = least_squares(func, x0=init_vals, args=(dist, mean, sigma))
    else:
        bounds = np.array(dist.params_support)[none_idx]
        bounds = list(zip(*bounds))
        if dist.__class__.__name__ in ["DiscreteWeibull"]:
            opt = least_squares(
                func, x0=init_vals, args=(dist, mean, sigma), bounds=bounds, loss="soft_l1"
            )
        else:
            opt = least_squares(func, x0=init_vals, args=(dist, mean, sigma), bounds=bounds)

    params = get_params(dist, opt["x"], none_idx, fixed)
    dist._parametrization(**params)
    return opt


def optimize_moments_rice(mean, std_dev):
    """
    Moment matching for the Rice distribution

    This function uses the Koay inversion technique, see: https://doi.org/10.1016/j.jmr.2006.01.016
    and https://en.wikipedia.org/wiki/Rice_distribution
    """

    ratio = mean / std_dev

    if ratio < 1.913:  # Rayleigh distribution
        nu = np.finfo(float).eps
        sigma = 0.655 * std_dev
    else:

        def xi(theta):
            return (
                2
                + theta**2
                - np.pi
                / 8
                * np.exp(-(theta**2) / 2)
                * ((2 + theta**2) * i0(theta**2 / 4) + theta**2 * i1(theta**2 / 4)) ** 2
            )

        def fpf(theta):
            return (xi(theta) * (1 + ratio**2) - 2) ** 0.5

        def func(theta):
            return np.abs(fpf(theta) - theta)

        theta = minimize(func, x0=fpf(1), bounds=[(0, None)]).x
        xi_theta = xi(theta)
        sigma = std_dev / xi_theta**0.5
        nu = (mean**2 + (xi_theta - 2) * sigma**2) ** 0.5

    return nu, sigma


def optimize_ml(dist, sample):
    def negll(params, dist, sample):
        dist._update(*params)
        if dist.kind == "continuous":
            neg = -dist.rv_frozen.logpdf(sample).sum()
        else:
            neg = -dist.rv_frozen.logpmf(sample).sum()
        return neg

    dist._fit_moments(np.mean(sample), np.std(sample))
    init_vals = dist.params

    opt = minimize(negll, x0=init_vals, bounds=dist.params_support, args=(dist, sample))

    dist._update(*opt["x"])

    return opt


def relative_error(dist, lower, upper, required_mass):
    if dist.kind == "discrete":
        lower -= 1
    computed_mass = dist.cdf(upper) - dist.cdf(lower)
    return abs((computed_mass - required_mass) / required_mass * 100), computed_mass


def get_distributions(dist_names):
    """
    Generate a subset of distributions which names agree with those in dist_names
    """
    dists = []
    for name in dist_names:
        dist = getattr(modules["preliz"], name)
        dists.append(dist())

    return dists


def fit_to_ecdf(selected_distributions, x_vals, ecdf, mean, std, x_min, x_max):
    """
    Minimize the difference between the cdf and the ecdf over a grid of values
    defined by x_min and x_max

    Note: This function is intended to be used with pz.roulette
    """
    fitted = Loss(len(selected_distributions))
    for dist in selected_distributions:
        if dist.__class__.__name__ == "BetaScaled":
            update_bounds_beta_scaled(dist, x_min, x_max)

        if dist._check_endpoints(x_min, x_max, raise_error=False):
            none_idx, fixed = get_fixed_params(dist)
            dist._fit_moments(mean, std)  # pylint:disable=protected-access
            loss = optimize_cdf(dist, x_vals, ecdf, none_idx, fixed)

            fitted.update(loss, dist)

    return fitted.dist


def fit_to_sample(selected_distributions, sample, x_min, x_max):
    """
    Maximize the likelihood given a sample
    """
    fitted = Loss(len(selected_distributions))
    sample_size = len(sample)
    for dist in selected_distributions:
        if dist.__class__.__name__ in ["BetaScaled", "TruncatedNormal"]:
            update_bounds_beta_scaled(dist, x_min, x_max)

        loss = np.inf
        if dist._check_endpoints(x_min, x_max, raise_error=False):
            dist._fit_mle(sample)  # pylint:disable=protected-access
            corr = get_penalization(sample_size, dist)
            if dist.kind == "continuous":
                loss = -(dist.rv_frozen.logpdf(sample).sum() - corr)
            else:
                loss = -(dist.rv_frozen.logpmf(sample).sum() - corr)

        fitted.update(loss, dist)

    return fitted


def update_bounds_beta_scaled(dist, x_min, x_max):
    dist.lower = x_min
    dist.upper = x_max
    dist.support = (x_min, x_max)
    return dist


def get_penalization(n, dist):
    """
    AIC with a correction for small sample sizes.

    Burnham, K. P.; Anderson, D. R. (2004),
    "Multimodel inference: understanding AIC and BIC in Model Selection"
    shorturl.at/IUWX6
    """
    k = len(dist.params)
    return k + ((k + 1) * k) / (-k + n - 1)


class Loss:
    def __init__(self, size=None):
        self.old_loss = np.inf
        self.dist = None
        self.count = 0
        self.distributions = np.empty(size, dtype=object)
        self.losses = np.empty(size)

    def update(self, loss, dist):
        self.distributions[self.count] = dist
        self.losses[self.count] = loss
        self.count += 1
        if loss < self.old_loss:
            self.old_loss = loss
            self.dist = dist


def get_fixed_params(distribution):
    none_idx = []
    fixed = []
    for idx, p_n in enumerate(distribution.param_names):
        value = getattr(distribution, p_n)
        if value is None:
            none_idx.append(idx)
        else:
            fixed.append(value)
    return none_idx, fixed
