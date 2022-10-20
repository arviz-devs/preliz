"""
Optimization routines and utilities
"""
from sys import modules
import numpy as np
from scipy.optimize import minimize, least_squares


def optimize_max_ent(dist, lower, upper, mass):
    def prob_bound(params, dist, lower, upper, mass):
        dist._update(*params)
        rv_frozen = dist.rv_frozen
        if dist.kind == "discrete":
            lower -= 1
        cdf0 = rv_frozen.cdf(lower)
        cdf1 = rv_frozen.cdf(upper)
        loss = (cdf1 - cdf0) - mass
        return loss

    def entropy_loss(params, dist):
        dist._update(*params)
        return -dist.rv_frozen.entropy()

    cons = {
        "type": "eq",
        "fun": prob_bound,
        "args": (dist, lower, upper, mass),
    }
    init_vals = dist.params
    bounds = dist.params_support
    if dist.name in ["halfstudent", "student"]:
        init_vals = init_vals[1:]
        bounds = bounds[1:]
    if dist.name == "skewnormal":
        init_vals = init_vals[:-1]
        bounds = bounds[:-1]
    if dist.name in ["betascaled", "truncatednormal"]:
        init_vals = init_vals[:-2]
        bounds = bounds[:-2]

    opt = minimize(entropy_loss, x0=init_vals, bounds=bounds, args=(dist), constraints=cons)
    dist._update(*opt["x"])

    return opt


def optimize_quartile(dist, x_vals):
    def func(params, dist, x_vals):
        dist._update(*params)
        loss = dist.rv_frozen.cdf(x_vals) - [0.25, 0.5, 0.75]
        return loss

    init_vals = dist.params
    if dist.name == "student":
        init_vals = init_vals[1:]
    if dist.name == "skewnormal":
        init_vals = init_vals[:-1]

    opt = least_squares(func, x0=init_vals, args=(dist, x_vals))
    dist._update(*opt["x"])

    return opt


def optimize_cdf(dist, x_vals, ecdf, **kwargs):
    def func(params, dist, x_vals, ecdf, **kwargs):
        dist._update(*params, **kwargs)
        loss = dist.rv_frozen.cdf(x_vals) - ecdf
        return loss

    init_vals = dist.params[:2]
    opt = least_squares(func, x0=init_vals, args=(dist, x_vals, ecdf), kwargs=kwargs)
    dist._update(*opt["x"])
    loss = opt["cost"]
    return loss


def optimize_matching_moments(dist, mean, sigma):
    def func(params, dist, mean, sigma):
        dist._update(*params)
        loss = ((dist.rv_frozen.mean() - mean) / mean) ** 2 + (
            (dist.rv_frozen.std() - sigma) / sigma
        ) ** 2
        return loss

    init_vals = dist.params
    opt = least_squares(func, x0=init_vals, args=(dist, mean, sigma))
    dist._update(*opt["x"])
    loss = opt["cost"]
    return loss


def optimize_ml(dist, sample):
    def negll(params, dist, sample):
        dist._update(*params)
        return -dist.rv_frozen.logpdf(sample).sum()

    dist._fit_moments(0, np.std(sample))
    init_vals = dist.params[::-1]

    opt = minimize(negll, x0=init_vals, bounds=dist.params_support, args=(dist, sample))

    dist._update(*opt["x"])

    return opt


def relative_error(dist, lower, upper, required_mass):
    if dist.kind == "discrete":
        lower -= 1
    computed_mass = dist.rv_frozen.cdf(upper) - dist.rv_frozen.cdf(lower)
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
    """
    fitted = Loss(len(selected_distributions))
    for dist in selected_distributions:
        kwargs = {}
        if dist.name == "betascaled":
            update_bounds_beta_scaled(dist, x_min, x_max)
            kwargs = {"lower": x_min, "upper": x_max}

        if dist._check_endpoints(x_min, x_max, raise_error=False):
            dist._fit_moments(mean, std)  # pylint:disable=protected-access
            loss = optimize_cdf(dist, x_vals, ecdf, **kwargs)

            fitted.update(loss, dist)

    return fitted.dist


def fit_to_sample(selected_distributions, sample, x_min, x_max):
    """
    Maximize the likelihood given a sample
    """
    fitted = Loss(len(selected_distributions))
    for dist in selected_distributions:
        if dist.name in ["betascaled", "truncatednormal"]:
            update_bounds_beta_scaled(dist, x_min, x_max)

        if dist._check_endpoints(x_min, x_max, raise_error=False):
            dist._fit_mle(sample)  # pylint:disable=protected-access
            if dist.kind == "continuous":
                loss = -dist.rv_frozen.logpdf(sample).sum()
            else:
                loss = -dist.rv_frozen.logpmf(sample).sum()

            fitted.update(loss, dist)

    return fitted


def update_bounds_beta_scaled(dist, x_min, x_max):
    dist.lower = x_min
    dist.upper = x_max
    dist.support = (x_min, x_max)
    return dist


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
