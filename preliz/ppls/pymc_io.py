"""Methods to communicate with PyMC."""

# pylint: disable=protected-access
from sys import modules

import numpy as np

try:
    from pytensor.tensor import vector, TensorConstant
    from pymc import logp, compile_pymc
    from pymc.util import is_transformed_name, get_untransformed_name
except ModuleNotFoundError:
    pass

from preliz.internal.optimization import get_distributions


def backfitting(prior, p_model, var_info2):
    """
    Fit the samples from prior into user provided model's prior.
    from the perspective of ppe "prior" is actually an approximated posterior
    but from the users perspective is its prior.
    We need to "backfitted" because we can not use arbitrary samples as priors.
    We need probability distributions.
    """
    new_priors = {}
    for key, size_inf in var_info2.items():
        if not size_inf[2]:
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


def compile_logp(model):
    """
    Compile the log-likelihood function for the model.
    We need to be able to condition it on parameters or data.
    Because during the optimization routine we need to change both.
    Currently this will fail for a prior that depends on other prior.
    """
    value = vector("value")
    rv_logp = logp(*model.observed_RVs, value)
    rv_logp_fn = compile_pymc([*model.free_RVs, value], rv_logp, on_unused_input="ignore")
    rv_logp_fn.trust_input = True

    def fmodel(params, obs, var_info, p_model):
        params = reshape_params(model, var_info, p_model, params)
        return -rv_logp_fn(*params, obs).sum()

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


def get_guess(model, free_rvs):
    """
    Get initial guess for optimization routine.
    """
    init = []

    free_rvs_names = [rv.name for rv in free_rvs]
    for key, value in model.initial_point().items():

        if is_transformed_name(key):
            name = get_untransformed_name(key)
            value = model.rvs_to_transforms[model.named_vars[name]].backward(value).eval()
        else:
            name = key

        if name in free_rvs_names:
            init.append(value)

    return np.concatenate([np.atleast_1d(arr) for arr in init]).flatten()


def get_model_information(model):  # pylint: disable=too-many-locals
    """
    Get information from the PyMC model.

    This needs some love. We even have a variable named var_info,
    and another one var_info2!
    """

    bounds = []
    prior = {}
    p_model = {}
    var_info = {}
    var_info2 = {}
    free_rvs = []
    pymc_to_preliz = get_pymc_to_preliz()
    rvs_to_values = model.rvs_to_values

    for r_v in model.free_RVs:
        if not non_constant_parents(r_v):
            free_rvs.append(r_v)

    for r_v in model.free_RVs:
        r_v_eval = r_v.eval()
        size = r_v_eval.size
        shape = r_v_eval.shape
        nc_parents = non_constant_parents(r_v)

        name = r_v.owner.op.name
        dist = pymc_to_preliz[name]
        p_model[r_v.name] = dist
        if nc_parents:
            idxs = [free_rvs.index(var_) for var_ in nc_parents]
            # the keys are the name of the (transformed) variable
            var_info[rvs_to_values[r_v].name] = (shape, size, idxs)
            # the keys are the name of the (untransformed) variable
            var_info2[r_v.name] = (shape, size, idxs)
        else:
            free_rvs.append(r_v)

            if size > 1:
                for i in range(size):
                    bounds.append(dist.support)
                    prior[f"{r_v.name}__{i}"] = []
            else:
                bounds.append(dist.support)
                prior[r_v.name] = []

            # the keys are the name of the (transformed) variable
            var_info[rvs_to_values[r_v].name] = (shape, size, nc_parents)
            # the keys are the name of the (untransformed) variable
            var_info2[r_v.name] = (shape, size, nc_parents)

    draws = model.observed_RVs[0].eval().size

    return bounds, prior, p_model, var_info, var_info2, draws, free_rvs


def write_pymc_string(new_priors, var_info):
    """
    Return a string with the new priors for the PyMC model.
    So the user can copy and paste, ideally with none to minimal changes.
    """

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


def reshape_params(model, var_info, p_model, params):
    """
    We flatten the parameters to be able to use them in the optimization routine.
    """
    size = 0
    value = []
    for var in model.value_vars:
        shape, new_size, idxs = var_info[var.name]
        if idxs:
            dist = p_model[var.name]
            dist._parametrization(*params[idxs])
            if new_size > 1:
                value.append(np.repeat(dist.mean(), new_size))
            else:
                value.append(dist.mean())
            size += new_size
        else:
            var_samples = params[size : size + new_size]
            value.append(var_samples.reshape(shape))
            size += new_size

    return value


def non_constant_parents(var_):
    """Find the parents of a variable that are not constant."""
    parents = []
    for variable in var_.get_parents()[0].inputs[3:]:
        if not isinstance(variable, TensorConstant):
            parents.append(variable)

    return parents
