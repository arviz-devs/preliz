from scipy.optimize import minimize


def optimize(dist, init_vals, lower, upper, mass):
    def prob_bound(params, dist, lower, upper, mass):
        dist._update(*params)  # pylint: disable=protected-access
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

    opt = minimize(
        dist._entropy_loss, x0=init_vals, constraints=cons  # pylint: disable=protected-access
    )
    return opt


def relative_error(dist, upper, lower, requiered_mass):
    if dist.kind == "discrete":
        lower -= 1
    computed_mass = dist.rv_frozen.cdf(upper) - dist.rv_frozen.cdf(lower)
    return abs((computed_mass - requiered_mass) / requiered_mass * 100)


def end_points_ints(lower, upper):
    return is_integer_num(lower) and is_integer_num(upper)


def is_integer_num(obj):
    if isinstance(obj, int):
        return True
    if isinstance(obj, float):
        return obj.is_integer()
    return False
