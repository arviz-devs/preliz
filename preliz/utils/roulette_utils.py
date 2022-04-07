import numpy as np

from ..distributions import all_continuous


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
            la = np.repeat((k / ncols * x_range) + x_min + ((x_range / ncols) / 2), v * 500 + 1)
            sample.extend(la)

    return sample, filled_columns


def get_distributions(cvars):
    """
    Generate a subset of scipy.stats distributions which names agrees with those in cvars
    """
    selection = [cvar.get() for cvar in cvars]
    dists = []
    for dist in all_continuous:
        if dist.__name__ in selection:
            dists.append(dist())

    return dists


# pylint: disable=unused-argument
def fit_to_sample(selected_distributions, sample, x_min, x_max, x_range):
    """
    Use a MLE approximated over a grid of values defined by x_min and x_max
    """
    x_vals = np.linspace(x_min, x_max, 500)
    sum_pdf_old = -np.inf
    fitted_dist = None
    for dist in selected_distributions:
        if x_min >= dist.dist.a and x_max <= dist.dist.b:
            dist.fit_mle(sample)
            # if dist.name == "beta_":
            #    dist.fit_mle(sample, floc=x_min, fscale=x_range)
            #    if (x_min < 0 or x_max > 1):
            #        dist.name = "scaled beta"
            # else:

            logpdf = dist.rv_frozen.logpdf(x_vals)
            sum_pdf = np.sum(logpdf[np.isfinite(logpdf)])
            if sum_pdf > sum_pdf_old:
                sum_pdf_old = sum_pdf
                fitted_dist = dist

    return fitted_dist
