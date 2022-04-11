# pylint: disable=protected-access
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

    cons = {
        "type": "eq",
        "fun": prob_bound,
        "args": (dist, lower, upper, mass),
    }
    init_vals = dist.params
    if dist.name == "student":
        init_vals = init_vals[1:]

    opt = minimize(dist._entropy_loss, x0=init_vals, constraints=cons)
    dist._update = opt["x"]

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
    dist._update = opt["x"]

    return opt


def relative_error(dist, lower, upper, required_mass):
    if dist.kind == "discrete":
        lower -= 1
    computed_mass = dist.rv_frozen.cdf(upper) - dist.rv_frozen.cdf(lower)
    return abs((computed_mass - required_mass) / required_mass * 100)


def end_points_ints(lower, upper):
    return is_integer_num(lower) and is_integer_num(upper)


def is_integer_num(obj):
    if isinstance(obj, int):
        return True
    if isinstance(obj, float):
        return obj.is_integer()
    return False
