import numpy as np
from scipy import stats


def weights_to_sample(weights, x_min, x_range, ncols):
    """
    Turn the weights (chips) into a sample over a grid.
    """

    sample = []
    filled_columns = 0

    if any(weights.values()):
        for k, v in weights.items():
            if v != 0:
                filled_columns += 1
            la = np.repeat((k / ncols * x_range) + x_min + ((x_range / ncols) / 2), v * 100 + 1)
            sample.extend(la)

    return sample, filled_columns


def get_scipy_distributions(cvars):
    """
    Generate a subset of scipy.stats distributions which names agrees with those in cvars
    """
    selection = [cvar.get() for cvar in cvars]
    scipy_dists = []
    for d in dir(stats):
        obj = getattr(stats, d)
        if hasattr(obj, "fit") and hasattr(obj, "name"):
            if obj.name in selection:
                scipy_dists.append(obj)
    return scipy_dists


def fit_to_sample(selected_distributions, sample, x_min, x_max, x_range):
    """
    Fit a sample to scipy distributions
    Use a MLE approximated over a grid of values defined by x_min and x_max
    """
    x_vals = np.linspace(x_min, x_max, 500)
    sum_pdf_old = -np.inf
    fitted_dist, can_dist, params, ref_pdf = [None] * 4
    for dist in selected_distributions:
        if dist.name == "beta":
            a, b, _, _ = dist.fit(sample, floc=x_min, fscale=x_range)
            can_dist = dist(a, b, loc=x_min, scale=x_range)
        elif dist.name == "norm":
            a, b = dist.fit(sample)
            can_dist = dist(a, b)
        elif dist.name == "gamma" and x_min >= 0:
            a, _, b = dist.fit(sample)
            can_dist = dist(a, scale=b)
            b = 1 / b
        elif dist.name == "lognorm" and x_min >= 0:
            b, _, a = dist.fit(sample)
            can_dist = dist(b, scale=a)
            a = np.log(a)

        if can_dist is not None:
            logpdf = can_dist.logpdf(x_vals)
            sum_pdf = np.sum(logpdf[np.isfinite(logpdf)])
            pdf = np.exp(logpdf)
            if sum_pdf > sum_pdf_old:
                sum_pdf_old = sum_pdf
                ref_pdf = pdf
                params = a, b
                fitted_dist = can_dist
                dist_name = dist.name
                if dist_name == "beta" and (x_min < 0 or x_max > 1):
                    dist_name = "scaled beta"
                fitted_dist.name = dist_name

    return fitted_dist, params, x_vals, ref_pdf
