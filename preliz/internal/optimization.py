"""
Optimization routines and utilities
"""
import warnings
from copy import copy

import numpy as np
from scipy.optimize import minimize, least_squares, root_scalar, brentq
from scipy.special import i0, i1, i0e, i1e  # pylint: disable=no-name-in-module
from .distribution_helper import init_vals as default_vals


def optimize_max_ent(dist, lower, upper, mass, none_idx, fixed_params, fixed_stat):
    def prob_bound(params, dist, lower, upper, mass):
        params = get_params(dist, params, none_idx, fixed_params)
        dist._parametrization(**params)
        if dist.kind == "discrete":
            lower -= 1
        cdf0 = dist.cdf(lower)
        cdf1 = dist.cdf(upper)
        loss = (cdf1 - cdf0) - mass
        if fixed_stat:
            loss = loss - abs(getattr(dist, fixed_stat[0])() - fixed_stat[1])
        return loss

    def entropy_loss(params, dist):
        params = get_params(dist, params, none_idx, fixed_params)
        dist._parametrization(**params)
        return -dist.entropy()

    cons = {
        "type": "eq",
        "fun": prob_bound,
        "args": (dist, lower, upper, mass),
    }
    init_vals = np.array(dist.params)[none_idx]
    bounds = np.array(dist.params_support)[none_idx]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Values in x were outside bounds")
        opt = minimize(entropy_loss, x0=init_vals, bounds=bounds, args=(dist), constraints=cons)

    params = get_params(dist, opt["x"], none_idx, fixed_params)
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


def optimize_pdf(dist, x_vals, epdf, none_idx, fixed):
    def func(params, dist, x_vals, epdf):
        params = get_params(dist, params, none_idx, fixed)
        dist._parametrization(**params)
        loss = dist.pdf(x_vals) - epdf
        return loss

    init_vals = np.array(dist.params)[none_idx]
    bounds = np.array(dist.params_support)[none_idx]
    bounds = list(zip(*bounds))

    opt = least_squares(func, x0=init_vals, args=(dist, x_vals, epdf), bounds=bounds)
    params = get_params(dist, opt["x"], none_idx, fixed)
    dist._parametrization(**params)
    loss = opt["cost"]
    return loss


def optimize_moments(dist, mean, sigma, params=None):
    def func(params, dist, mean, sigma):
        params = get_params(dist, params, none_idx, fixed)
        dist._parametrization(**params)
        loss = abs(dist.mean() - mean) + abs(dist.std() - sigma)
        return loss

    none_idx, fixed = get_fixed_params(dist)

    if params is not None:
        dist._update(*params)
    else:
        name = dist.__class__.__name__
        if name == "Truncated":
            vals = copy(default_vals["Truncated"])
            vals.update(default_vals[dist.dist.__class__.__name__])
            dist._parametrization(**vals)
        else:
            dist._update(**default_vals[name])

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
        return dist._neg_logpdf(sample)

    dist._fit_moments(np.mean(sample), np.std(sample))
    init_vals = dist.params

    opt = minimize(negll, x0=init_vals, bounds=dist.params_support, args=(dist, sample))

    dist._update(*opt["x"])

    return opt


def optimize_dirichlet_mode(lower_bounds, mode, target_mass, _dist):
    def prob_approx(tau, lower_bounds, mode, _dist):
        alpha = [1 + tau * mode_i for mode_i in mode]
        a_0 = sum(alpha)
        marginal_prob_list = []
        for a_i, lbi in zip(alpha, lower_bounds):
            _dist._parametrization(a_i, a_0 - a_i)
            marginal_prob_list.append(_dist.cdf(lbi))

        mean_cdf = np.mean(marginal_prob_list)
        return mean_cdf, alpha

    tau = 1
    new_prob, alpha = prob_approx(tau, lower_bounds, mode, _dist)

    while abs(new_prob - target_mass) > 0.0001:
        if new_prob < target_mass:
            tau -= 0.5 * tau
        else:
            tau += 0.5 * tau

        new_prob, alpha = prob_approx(tau, lower_bounds, mode, _dist)

    return new_prob, alpha


def optimize_beta_mode(lower, upper, tau_not, mode, dist, mass, prob):
    while abs(prob - mass) > 0.0001:
        alpha = 1 + mode * tau_not
        beta = 1 + (1 - mode) * tau_not
        dist._parametrization(alpha, beta)
        prob = dist.cdf(upper) - dist.cdf(lower)

        if prob > mass:
            tau_not -= 0.5 * tau_not
        else:
            tau_not += 0.5 * tau_not


def optimize_hdi(dist, mass):
    def interval_loss(params):
        cdf = dist.cdf(params)
        loss = (cdf[1] - cdf[0] + mass_at_boundaries) - mass
        return loss

    def interval_short(params):
        if params[1] < params[0]:
            return np.inf
        return params[1] - params[0]

    cons = {
        "type": "eq",
        "fun": interval_loss,
    }
    lower, upper = dist.support
    mass_at_boundaries = dist.pdf(lower) + dist.pdf(upper)
    init_vals = dist.eti(mass=mass, fmt="none")
    bounds = [(lower, upper)]
    opt = minimize(interval_short, x0=init_vals, bounds=bounds, constraints=cons)

    lower, upper = opt.x
    if dist.kind == "discrete":
        upper = np.floor(upper - 1).astype(int)
        lower = np.ceil(lower).astype(int)

    return lower, upper


def optimize_pymc_model(
    fmodel,
    target,
    num_draws,
    opt_iterations,
    initial_guess,
    rng,
):
    prior_array = np.zeros((opt_iterations, len(initial_guess)))

    for idx in range(opt_iterations + 1):
        # can we sample systematically from these and less random?
        # This should be more flexible and allow other targets than just
        # a PreliZ distribution
        if isinstance(target, list):
            obs = get_weighted_rvs(target, num_draws, rng)
        else:
            obs = target.rvs(num_draws, idx)
        result = minimize(
            fmodel,
            initial_guess,
            tol=0.001,
            method="powell",
            args=(obs),
        )
        # To help minimize the effect of priors we don't save the first result
        # and instead we use it as the initial guess for the next optimization
        # Updating the initial guess also helps reduce computational times
        if idx:
            prior_array[idx - 1] = result.x
        initial_guess = result.x

    return prior_array


def relative_error(dist, lower, upper, required_mass):
    if dist.kind == "discrete":
        lower -= 1
    computed_mass = dist.cdf(upper) - dist.cdf(lower)
    return abs((computed_mass - required_mass) / required_mass * 100), computed_mass


def fit_to_epdf(selected_distributions, x_vals, epdf, mean, std, x_min, x_max, extra_pros):
    """
    Minimize the difference between the pdf and the epdf over a grid of values
    defined by x_min and x_max

    Note: This function is intended to be used with pz.roulette
    """
    fitted = Loss(len(selected_distributions))
    for dist in selected_distributions:
        if dist.__class__.__name__ in extra_pros:
            try:
                dist._parametrization(**extra_pros[dist.__class__.__name__])
            except TypeError:
                pass
        if dist.__class__.__name__ == "BetaScaled":
            update_bounds_beta_scaled(dist, x_min, x_max)

        if dist._check_endpoints(x_min, x_max, raise_error=False):
            none_idx, fixed = get_fixed_params(dist)
            dist._fit_moments(mean, std)  # pylint:disable=protected-access
            loss = optimize_pdf(dist, x_vals, epdf, none_idx, fixed)

            fitted.update(loss, dist)

    return fitted.dist


def fit_to_sample(selected_distributions, sample, x_min, x_max):
    """
    Maximize the likelihood given a sample
    """
    fitted = Loss(len(selected_distributions))
    for dist in selected_distributions:  # pylint: disable=too-many-nested-blocks
        if dist.__class__.__name__ in ["BetaScaled", "TruncatedNormal"]:
            update_bounds_beta_scaled(dist, x_min, x_max)

        loss = np.inf
        if dist._check_endpoints(x_min, x_max, raise_error=False):
            if sample.ndim > 1:
                dists = []
                neg_logpdf = 0
                for s in sample:
                    dist_i = copy(dist)
                    dist_i._fit_mle(s)
                    neg_logpdf += dist_i._neg_logpdf(s)
                    dists.append(dist_i)
                new_dict = {}
                for d in dists:
                    params = d.params_dict
                    for k, v in params.items():
                        if k in new_dict:
                            new_dict[k].append(v)
                        else:
                            new_dict[k] = [v]

                dist._parametrization(**{k: np.asarray(v) for k, v in new_dict.items()})
            else:
                dist._fit_mle(sample)  # pylint:disable=protected-access
                neg_logpdf = dist._neg_logpdf(sample)
            corr = get_penalization(sample.size, dist)
            loss = neg_logpdf + corr
        fitted.update(loss, dist)

    return fitted


def fit_to_quartile(selected_distributions, q1, q2, q3, extra_pros):
    error = np.inf
    fitted_dist = None

    for distribution in selected_distributions:
        if distribution.__class__.__name__ in extra_pros:
            distribution._parametrization(**extra_pros[distribution.__class__.__name__])
            if distribution.__class__.__name__ == "BetaScaled":
                update_bounds_beta_scaled(
                    distribution,
                    extra_pros[distribution.__class__.__name__]["lower"],
                    extra_pros[distribution.__class__.__name__]["upper"],
                )
        if distribution._check_endpoints(q1, q3, raise_error=False):
            none_idx, fixed = get_fixed_params(distribution)

            distribution._fit_moments(
                mean=q2, sigma=(q3 - q1) / 1.35
            )  # pylint:disable=protected-access

            optimize_quartile(distribution, (q1, q2, q3), none_idx, fixed)

            r_error, _ = relative_error(distribution, q1, q3, 0.5)
            if r_error < error:
                fitted_dist = distribution
                error = r_error

    return fitted_dist


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
    https://doi.org/10.1177/0049124104268
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


def find_kappa(data, mu):
    ere = np.mean(np.cos(mu - data))

    if ere > 0:

        def solve_for_kappa(kappa):
            return i1e(kappa) / i0e(kappa) - ere

        root_res = root_scalar(
            solve_for_kappa, method="brentq", bracket=(np.finfo(float).tiny, 1e16)
        )
        return root_res.root
    else:
        return np.finfo(float).tiny


def find_ppf(dist, q):
    q = np.atleast_1d(q)
    ppf = np.zeros_like(q)
    lower, upper = dist.support
    for idx, q_i in enumerate(q):
        if q_i < 0:
            ppf[idx] = np.nan
        elif q_i > 1:
            ppf[idx] = np.nan
        elif q_i == 0:
            if dist.kind == "discrete":
                ppf[idx] = lower - 1
            else:
                ppf[idx] = lower
        elif q_i == 1:
            ppf[idx] = upper
        else:
            if dist.__class__.__name__ in ["HyperGeometric", "BetaBinomial"]:
                ppf[idx] = _ppf_single(dist, q_i) + 1
            else:
                ppf[idx] = _ppf_single(dist, q_i)
    return ppf[0] if len(ppf) == 1 else ppf


def _ppf_single(dist, q):
    def func(x, dist, q):
        return dist.cdf(x) - q

    factor = 10.0
    left, right = dist.support

    left = min(-factor, right)
    while func(left, dist, q) > 0.0:
        left, right = left * factor, left

    right = max(factor, left)
    while func(right, dist, q) < 0.0:
        left, right = right, right * factor

    return brentq(func, left, right, args=(dist, q))


def get_weighted_rvs(target, size, rng):
    targets = [t[0] for t in target]
    weights = [t[1] for t in target]
    target_rnd_choices = np.random.choice(len(targets), size=size, p=weights)
    samples = [target.rvs(size, random_state=rng) for target in targets]
    return np.choose(target_rnd_choices, samples)


def _root(n_p, k_sq, a_sq, x):
    def _fixed_point(t_s, x_len, k_sq, a_sq):
        """Calculate t-zeta*gamma^[l](t).

        Implementation of the function t-zeta*gamma^[l](t) derived from equation (30) in [1].

        References
        ----------
        .. [1] Kernel density estimation via diffusion.
        Z. I. Botev, J. F. Grotowski, and D. P. Kroese.
        Ann. Statist. 38 (2010), no. 5, 2916--2957.
        """
        k_sq = np.asarray(k_sq, dtype=np.float64)
        a_sq = np.asarray(a_sq, dtype=np.float64)

        k_o = 7
        func = np.sum(np.power(k_sq, k_o) * a_sq * np.exp(-k_sq * np.pi**2 * t_s))
        func *= 0.5 * np.pi ** (2.0 * k_o)

        for ite in np.arange(k_o - 1, 2 - 1, -1):
            c_1 = (1 + 0.5 ** (ite + 0.5)) / 3
            c_2 = np.prod(np.arange(1.0, 2 * ite + 1, 2, dtype=np.float64))
            c_2 /= (np.pi / 2) ** 0.5
            t_j = np.power((c_1 * (c_2 / (x_len * func))), (2.0 / (3.0 + 2.0 * ite)))
            func = np.sum(k_sq**ite * a_sq * np.exp(-k_sq * np.pi**2.0 * t_j))
            func *= 0.5 * np.pi ** (2 * ite)

        out = t_s - (2 * x_len * np.pi**0.5 * func) ** (-0.4)
        return out

    # The right bound is at most 0.01
    found = False
    n_p = max(min(1050, n_p), 50)
    tol = 10e-12 + 0.01 * (n_p - 50) / 1000

    while not found:
        try:
            band_w, res = brentq(
                _fixed_point, 0, 0.01, args=(n_p, k_sq, a_sq), full_output=True, disp=False
            )
            found = res.converged
        except ValueError:
            band_w = 0
            tol *= 2.0
            found = False
        if band_w <= 0 or tol >= 1:
            band_w = (_bw_silverman(x) / np.ptp(x)) ** 2
            return band_w
    return band_w


def _bw_silverman(x, x_std=None):  # pylint: disable=unused-argument
    """Silverman's Rule."""
    x_std = np.std(x)
    q75, q25 = np.percentile(x, [75, 25])
    x_iqr = q75 - q25
    a = min(x_std, x_iqr / 1.34)
    band_w = 0.9 * a * len(x) ** (-0.2)
    return band_w
