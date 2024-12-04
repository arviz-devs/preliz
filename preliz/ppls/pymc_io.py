"""Methods to communicate with PyMC."""

# pylint: disable=protected-access
from copy import copy
from sys import modules

import numpy as np

try:
    from pytensor.tensor import matrix, TensorConstant
    from pytensor import function
    from pytensor.graph.basic import ancestors
    from pymc.pytensorf import compile_pymc, join_nonshared_inputs
    from pymc.util import is_transformed_name, get_untransformed_name
except ModuleNotFoundError:
    pass

from preliz.internal.distribution_helper import get_distributions


def back_fitting_pymc(prior, preliz_model, var_info):
    """
    Fit the samples from prior into user provided model's prior.
    from the perspective of ppe "prior" is actually an approximated posterior
    but from the users perspective is its prior.
    We need to "backfit" because we can not use arbitrary samples as priors.
    We need probability distributions.
    """
    new_priors = {}
    for rv_name, (_, size, *_) in var_info.items():
        if size > 1:
            params = []
            for i in range(size):
                opt_values = prior[rv_name][:, i]
                dist = preliz_model[rv_name]
                dist._fit_mle(opt_values)
                params.append(dist.params)
            dist._parametrization(*[np.array(x) for x in zip(*params)])
        else:
            opt_values = prior[rv_name]
            dist = preliz_model[rv_name]
            dist._fit_mle(opt_values)

        new_priors[rv_name] = dist

    return new_priors


def compile_mllk(model):
    """
    Compile the log-likelihood function for the model to be able to condition on both
    data and parameters.
    """
    obs_rvs = model.observed_RVs[0]
    old_y_value = model.rvs_to_values[obs_rvs]
    new_y_value = obs_rvs.type()
    model.rvs_to_values[obs_rvs] = new_y_value

    vars_ = model.value_vars
    initial_point = model.initial_point()

    [logp], raveled_inp = join_nonshared_inputs(
        point=initial_point, outputs=[model.datalogp], inputs=vars_
    )

    rv_logp_fn = compile_pymc([raveled_inp, new_y_value], logp)
    rv_logp_fn.trust_input = True

    def fmodel(params, obs):
        return -rv_logp_fn(params, obs).sum()

    return fmodel, old_y_value, obs_rvs


def get_initial_guess(model):
    """
    Get initial guess for optimization routine.
    """
    return np.concatenate([np.ravel(value) for value in model.initial_point().values()])


def extract_preliz_distributions(model):
    """
    Extract the corresponding PreliZ distributions from a PyMC model

    Parameters
    ----------
    model : a PyMC model

    Returns
    -------
    preliz_model : a dictionary of RVs names as keys and PreliZ distributions as values
    num_draws : the sample size of the observed
    """
    all_distributions = [
        dist
        for dist in modules["preliz.distributions"].__all__
        if dist not in ["Truncated", "Censored", "Hurdle", "Mixture"]
    ]
    pymc_to_preliz = dict(
        zip([dist.lower() for dist in all_distributions], get_distributions(all_distributions)),
    )

    preliz_model = {}
    for r_v in model.free_RVs:
        dist_name = (
            r_v.owner.op.name if r_v.owner.op.name else str(r_v.owner.op).split("RV", 1)[0].lower()
        )
        dist = copy(pymc_to_preliz[dist_name])
        preliz_model[r_v.name] = dist

    return preliz_model


def retrieve_variable_info(model):
    """
    Get the shape, size, transformation and parents of each free random variable in a PyMC model.
    """

    var_info = {}
    initial_point = model.initial_point()
    for v_var in model.value_vars:
        name = v_var.name
        rvs = model.values_to_rvs[v_var]
        nc_parents = non_constant_parents(rvs, model)
        idx_parents = []
        if nc_parents:
            idx_parents = [model.free_RVs.index(var_) for var_ in nc_parents]

        if is_transformed_name(name):
            name = get_untransformed_name(name)
            x_var = matrix(f"{name}_transformed")
            z_var = model.rvs_to_transforms[rvs].backward(x_var)
            transformation = function(inputs=[x_var], outputs=z_var)
        else:
            transformation = None

        var_info[name] = (
            initial_point[v_var.name].shape,
            initial_point[v_var.name].size,
            transformation,
            idx_parents,
        )

    num_draws = model.observed_RVs[0].eval().size

    return var_info, num_draws


def unravel_projection(prior_array, var_info, iterations):
    size = 0
    prior_dict = {}
    for key, values in var_info.items():
        shape, new_size, transformation, _ = values
        vector = prior_array[:, size : size + new_size]
        if transformation is not None:
            vector = transformation(vector)
        prior_dict[key] = vector.reshape(iterations, *shape).squeeze()
        size += new_size

    return prior_dict


def write_pymc_string(new_priors, var_info):
    """
    Return a string with the new priors for the PyMC model.
    So the user can copy and paste, ideally with none to minimal changes.
    """

    header = "with pm.Model() as model:\n"
    variables = []
    names = list(new_priors.keys())
    for key, value in new_priors.items():
        idxs = var_info[key][-1]
        if idxs:
            for i in idxs:
                nkey = names[i]
                cp_dist = copy(new_priors[nkey])
                cp_dist._fit_moments(np.mean(value.mean()), np.mean(value.std()))

                dist_name, dist_params = repr(cp_dist).split("(")
                size = var_info[nkey][1]
                if size > 1:
                    dist_params = dist_params.split(")")[0]
                    variables[
                        i
                    ] = f'    {nkey:} = pm.{dist_name}("{nkey}", {dist_params}, shape={size})\n'
                else:
                    variables[i] = f'    {nkey:} = pm.{dist_name}("{nkey}", {dist_params}\n'

        else:
            dist_name, dist_params = repr(value).split("(")
            size = var_info[key][1]
            if size > 1:
                dist_params = dist_params.split(")")[0]
                variables.append(
                    f'    {key:} = pm.{dist_name}("{key}", {dist_params}, shape={size})\n'
                )
            else:
                variables.append(f'    {key:} = pm.{dist_name}("{key}", {dist_params}\n')

    return "".join([header] + variables)


def non_constant_parents(rvs, model):
    """Find the parents of a variable that are not constant."""
    parents = []
    for variable in rvs.get_parents()[0].inputs[2:]:
        if not isinstance(variable, TensorConstant):
            for free_rv in model.free_RVs:
                if free_rv in list(ancestors([variable])) and free_rv not in parents:
                    parents.append(free_rv)
    return parents
