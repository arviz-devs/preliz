"""
Optimization routines and utilities
"""
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
    if dist.name == "student":
        init_vals = init_vals[1:]
        bounds = bounds[1:]

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

    opt = least_squares(func, x0=init_vals, args=(dist, x_vals))
    dist._update(*opt["x"])

    return opt


def optimize_cdf(dist, x_vals, pcdf):
    def func(params, dist, x_vals, pcdf):
        dist._update(*params)
        loss = dist.rv_frozen.cdf(x_vals) - pcdf
        return loss

    init_vals = dist.params
    opt = least_squares(func, x0=init_vals, args=(dist, x_vals, pcdf))
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


def relative_error(dist, lower, upper, required_mass):
    if dist.kind == "discrete":
        lower -= 1
    computed_mass = dist.rv_frozen.cdf(upper) - dist.rv_frozen.cdf(lower)
    return abs((computed_mass - required_mass) / required_mass * 100), computed_mass
