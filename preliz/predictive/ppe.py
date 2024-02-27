from sys import modules

import numpy as np
from scipy.optimize import minimize


from pytensor.tensor import vector as pt_vector
from pymc import logp, compile_pymc
from pymc.util import is_transformed_name, get_untransformed_name

from ..internal.optimization import get_distributions


def ppe(model, target):
    """
    Projective Predictive Elicitation
    """
    bounds, prior, p_model, var_info, var_info2 = get_model_information(model)
    guess = get_guess(model)
    fmodel = compile_logp(model)

    prior = optimize_pymc_model(fmodel, target, prior, guess, bounds, var_info)
    new_priors = backfitting(prior, p_model, var_info2)

    return prior, write_pymc_string(new_priors, var_info2)


def backfitting(prior, p_model, var_info2):  #### we already have a function with this name
    new_priors = {}
    for key, size_inf in var_info2.items():
        size = size_inf[1]
        if size > 1:
            params = []
            for i in range(size):
                value = prior[f"{key}__{i}"]
                dist = p_model[key]
                dist._fit_mle(value)
                params.append(dist.params)
            dist._parametrization(*[np.array(x) for x in zip(*params)])
        else:
            value = prior[key]
            dist = p_model[key]
            dist._fit_mle(value)

        new_priors[key] = dist

    return new_priors


## some pymc to preliz module???
def compile_logp(model):
    value = pt_vector("value")
    rv_logp = logp(*model.observed_RVs, value)
    rv_logp_fn = compile_pymc([*model.free_RVs, value], rv_logp)

    def fmodel(params, obs, var_info):
        params = reshape_params(model, var_info, params)
        y = -rv_logp_fn(*params, obs).sum()
        return y

    return fmodel


def get_pymc_to_preliz():
    """
    Generate dictionary mapping pymc to preliz distributions
    """
    all_distributions = modules["preliz.distributions"].__all__
    pymc_to_preliz = dict(
        zip([dist.lower() for dist in all_distributions], get_distributions(all_distributions))
    )
    return pymc_to_preliz


def get_guess(model):
    init = []
    for key, value in model.initial_point().items():
        if is_transformed_name(key):
            un_name = get_untransformed_name(key)
            value = model.rvs_to_transforms[model.named_vars[un_name]].backward(value).eval()

        init.append(value)

    return np.concatenate([np.atleast_1d(arr) for arr in init]).flatten()


def get_model_information(model):
    bounds = []
    prior = {}
    p_model = {}
    var_info = {}
    var_info2 = {}
    pymc_to_preliz = get_pymc_to_preliz()
    rvs_to_values = model.rvs_to_values
    for r_v in model.free_RVs:
        name = r_v.owner.op.name
        dist = pymc_to_preliz[name]
        r_v_eval = r_v.eval()
        size = r_v_eval.size
        shape = r_v_eval.shape
        if size > 1:
            for i in range(size):
                bounds.append(dist.support)
                prior[f"{r_v.name}__{i}"] = []
        else:
            bounds.append(dist.support)
            prior[r_v.name] = []

        var_info[rvs_to_values[r_v].name] = (shape, size)
        var_info2[r_v.name] = (shape, size)

        p_model[r_v.name] = dist

    return bounds, prior, p_model, var_info, var_info2


def write_pymc_string(new_priors, var_info):
    header = "with pm.Model() as model:\n"

    for key, value in new_priors.items():
        dist_name, dist_params = repr(value).split("(")
        size = var_info[key][1]
        if size > 1:
            dist_params = dist_params.split(")")[0]
            header += f'{key:>4} = pm.{dist_name}("{key}", {dist_params}, shape={size})\n'
        else:
            header += f'{key:>4} = pm.{dist_name}("{key}", {dist_params}\n'

    return header


def reshape_params(model, var_info, params):
    size = 0
    value = []
    for var in model.value_vars:
        shape, new_size = var_info[var.name]
        var_samples = params[size : size + new_size]
        # Round discrete variable samples. The rounded values were the ones
        # actually used in the logp evaluations (see logp_forw)
        # if var.dtype in discrete_types:
        #    var_samples = np.round(var_samples).astype(var.dtype)
        value.append(np.atleast_1d(var_samples).reshape(shape))
        size += new_size
    return value


### optimization
def optimize_pymc_model(fmodel, target, prior, initial_guess, bounds, var_info):
    for _ in range(1000):
        obs = target.rvs(120)  # can we sample systematically from these and less random?
        result = minimize(
            fmodel,
            initial_guess,
            tol=0.001,
            method="SLSQP",
            args=(obs, var_info),
            bounds=bounds,
        )

        optimal_params = result.x
        initial_guess = optimal_params

        for key, param in zip(prior.keys(), optimal_params):
            prior[key].append(param)

    # convert to numpy arrays
    for key, value in prior.items():
        prior[key] = np.array(value)

    return prior
