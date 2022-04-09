import numpy as np
from scipy.optimize import least_squares


from ..distributions import all_continuous


def weights_to_ecdf(weights, x_min, x_range, ncols):
    """
    Turn the weights (chips) into the empirical cdf
    """
    filled_columns = 0
    x_vals = []
    pcdf = []
    cum_sum = 0

    values = list(weights.values())
    mean = np.mean(values)
    std = np.std(values)
    total = sum(values) + ncols
    if any(weights.values()):
        for k, v in weights.items():
            if v != 0:
                filled_columns += 1
            x_val = (k / ncols * x_range) + x_min + ((x_range / ncols))
            x_vals.append(x_val)
            cum_sum += v / total
            pcdf.append(cum_sum)

    return x_vals, pcdf, mean, std, filled_columns


def get_distributions(cvars):
    """
    Generate a subset of distributions which names agrees with those in cvars
    """
    selection = [cvar.get() for cvar in cvars]
    dists = []
    for dist in all_continuous:
        if dist.__name__ in selection:
            dists.append(dist())

    return dists


def fit_to_ecdf(selected_distributions, x_vals, pcdf, mean, std, x_min, x_max):
    """
    Use a MLE approximated over a grid of values defined by x_min and x_max
    """
    loss_old = np.inf
    fitted_dist = None
    for dist in selected_distributions:
        if x_min >= dist.dist.a and x_max <= dist.dist.b:
            dist.fit_moments(mean, std)
            init_vals = dist.params
            opt = least_squares(
                func, x0=init_vals, args=(dist, x_vals, pcdf)
            )  # pylint: disable=protected-access
            dist._update(*opt["x"])
            loss = opt["cost"]

            if loss < loss_old:
                loss_old = loss
                fitted_dist = dist

    return fitted_dist


def func(params, dist, x_vals, pcdf):
    dist._update(*params)  # pylint: disable=protected-access
    loss = dist.rv_frozen.cdf(x_vals) - pcdf
    return loss
