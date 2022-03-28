from scipy.optimize import least_squares


def optimize(dist, init_vals, lower, upper, mass):
    opt = least_squares(dist._cdf_loss, x0=init_vals, args=(lower, upper, mass), method="dogbox")
    return opt


def relative_error(rv_frozen, upper, lower, requiered_mass):
    computed_mass = rv_frozen.cdf(upper) - rv_frozen.cdf(lower)
    return (computed_mass - requiered_mass) / requiered_mass * 100
