"""Methods to communicate with PyMC."""

import warnings
from copy import copy
from sys import modules

import numpy as np

try:
    from pymc.pytensorf import compile, join_nonshared_inputs
    from pymc.util import get_untransformed_name, is_transformed_name
    from pytensor import function
    from pytensor.graph.traversal import ancestors
    from pytensor.tensor import TensorConstant, matrix
except ImportError:
    warnings.warn("PyMC not installed. PyMC related functions will not work.")

from preliz.internal.distribution_helper import get_distributions


def back_fitting_pymc(prior, preliz_model, var_info, new_families=None):
    """
    Fit the samples from prior into user provided model's prior.

    From the perspective of ppe "prior" is actually an approximated posterior
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
                # Not sure how to fit alternative families.
                dist = preliz_model[rv_name]
                dist._fit_mle(opt_values)
                params.append(dist.params)
            dist._parametrization(*[np.array(x) for x in zip(*params)])
        else:
            opt_values = prior[rv_name]
            dists = set_families(preliz_model[rv_name], rv_name, new_families)
            mle = getattr(modules["preliz.unidimensional"], "mle")
            idx, _ = mle(dists, opt_values, plot=False)
            dist = dists[idx[0]]

        new_priors[rv_name] = dist

    return new_priors


def set_families(dist, var, new_families):
    dists = [dist]
    if new_families is not None:
        if new_families == "auto":
            alt = [
                getattr(modules["preliz.distributions"], d)
                for d in ["Normal", "HalfNormal", "Gamma"]
            ]
            dists += [a for a in alt if dist.__class__.__name__ != a.__class__.__name__]
        elif isinstance(new_families, list):
            dists += new_families
        elif isinstance(new_families, dict):
            dists += new_families.get(var, [])

    return dists


def compile_mllk(model):
    """
    Compile the log-likelihood for a pymc model.

    The compiled function allow us to condition on both data and parameters.
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

    rv_logp_fn = compile([raveled_inp, new_y_value], logp)
    rv_logp_fn.trust_input = True

    def fmodel(params, obs):
        return -rv_logp_fn(params, obs).sum()

    return fmodel, old_y_value, obs_rvs


def get_initial_guess(model):
    """Get initial guess for optimization routine."""
    return np.concatenate([np.ravel(value) for value in model.initial_point().values()])


def extract_preliz_distributions(model):
    """
    Extract the corresponding PreliZ distributions from a PyMC model.

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
    """Get shape, size, transformation and parents of each free RV in a PyMC model."""
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
                    # fmt: off
                    variables[i] = f'    {nkey:} = pm.{dist_name}("{nkey}", {dist_params}, shape={size})\n' # noqa: E501
                    # fmt: on
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


def if_pymc_get_preliz(dist):
    """Check if dist is a PyMC distribution and if so convert to PreliZ."""
    if dist.__class__.__name__ == "TensorVariable":
        dist = from_pymc(dist)
    return dist


def from_pymc(dist):
    """Convert a PyMC distribution to a PreliZ distribution.

    Parameters
    ----------
    dist : PyMC distribution

    Returns
    -------
    PreliZ distribution
    """
    name = dist.owner.op._print_name[0]
    if name == "MultivariateNormal":
        name = "MvNormal"

    if name == "Censored":
        base_dist = dist.owner.inputs[0]
        lower = _as_scalar(dist.owner.inputs[1].eval())
        upper = _as_scalar(dist.owner.inputs[2].eval())
        if np.isnan(lower):
            lower = -np.inf
        if np.isnan(upper):
            upper = np.inf

        BaseDist = from_pymc(base_dist)
        return modules["preliz.distributions"].Censored(BaseDist, lower=lower, upper=upper)

    if "Truncated" in name:
        base_dist_name = name.split("Truncated")[1]
        base_params = [v.eval() for v in dist.owner.inputs[2:-2]]
        lower = _as_scalar(dist.owner.inputs[-2].eval())
        upper = _as_scalar(dist.owner.inputs[-1].eval())
        if np.isnan(lower):
            lower = -np.inf
        if np.isnan(upper):
            upper = np.inf

        BaseDist = getattr(modules["preliz.distributions"], base_dist_name)
        return modules["preliz.distributions"].Truncated(
            _reparametrize(BaseDist, base_dist_name, base_params), lower=lower, upper=upper
        )

    elif name == "Mixture":
        name_0 = dist.owner.inputs[2].owner.op._print_name[0]
        if name_0 == "DiracDelta":
            base_node = dist.owner.inputs[-1]
            base_name = base_node.owner.op._print_name[0]
            base_params = [v.eval() for v in base_node.owner.inputs[2:]]
            psi = _nan_to_none(dist.owner.inputs[1].eval())[1]
            ZeroInflated = getattr(modules["preliz.distributions"], f"ZeroInflated{base_name}")
            if base_name == "NegativeBinomial":
                n, p = base_params
                mu = n * (1 - p) / p
                base_params = [mu, n]

            base_params = _nan_to_none(base_params)
            return ZeroInflated(psi, *base_params)

        else:
            components = dist.owner.inputs[2:]
            weights = _nan_to_none(dist.owner.inputs[1].eval())
            PreliZ_components = [from_pymc(comp) for comp in components]
            return modules["preliz.distributions"].Mixture(PreliZ_components, weights=weights)

    elif name == "Hurdle":
        base_type_name = dist.owner.inputs[-1].owner.op._print_name[0].replace("Truncated", "")
        psi = _nan_to_none(dist.owner.inputs[1].eval())[-1]
        base_params = _nan_to_none([v.eval() for v in dist.owner.inputs[-1].owner.inputs[2:]])
        BaseDist = getattr(modules["preliz.distributions"], base_type_name)(*base_params)
        return getattr(modules["preliz.distributions"], "Hurdle")(BaseDist, psi)

    else:
        if name in ["HalfNormal", "HalfCauchy"]:
            params_inputs = [v.eval() for v in dist.owner.inputs[3:]]
        else:
            params_inputs = [v.eval() for v in dist.owner.inputs[2:]]

        try:
            Dist = getattr(modules["preliz.distributions"], name)
        except AttributeError:
            raise NotImplementedError(f"No PreliZ distribution named {name}")

        return _reparametrize(Dist, name, params_inputs)


def _as_scalar(x):
    x = np.asarray(x)
    return x.item() if x.shape == () or x.size == 1 else x


def _reparametrize(Dist, name, params_inputs):
    if name == "Exponential":
        lamda_ = _nan_to_none(1 / params_inputs[0])
        return Dist(lamda_)
    if name == "Gamma":
        alpha, inv_beta = params_inputs
        alpha, beta = _nan_to_none((alpha, 1 / inv_beta))
        return Dist(alpha=alpha, beta=beta)
    if name == "Rice":
        b, sigma = params_inputs
        nu, sigma = _nan_to_none((b * sigma, sigma))
        return Dist(nu=nu, sigma=sigma)
    if name == "SkewNormal":
        alpha, mu, sigma = _nan_to_none(params_inputs)
        return Dist(alpha=alpha, mu=mu, sigma=sigma)
    if name == "Triangular":
        lower, upper, c = _nan_to_none(params_inputs)
        return Dist(lower=lower, c=c, upper=upper)
    if name == "Wald":
        mu, lam, _ = _nan_to_none(params_inputs)
        return Dist(mu=mu, lam=lam)
    if name == "BetaBinomial":
        n, alpha, beta = _nan_to_none(params_inputs)
        return Dist(alpha=alpha, beta=beta, n=n)
    if name == "NegativeBinomial":
        n, p = params_inputs
        mu, n = _nan_to_none((n * (1 - p) / p, n))
        return Dist(mu=mu, alpha=n)
    if name == "HyperGeometric":
        good, bad, n = params_inputs
        N, good, n = _nan_to_none((good + bad, good, n))
        return Dist(N=N, k=good, n=n)

    return Dist(*_nan_to_none(params_inputs))


def _nan_to_none(params):
    if isinstance(params, (float, int, np.integer, np.floating)):
        return None if np.isnan(params) else params
    return [None if np.isnan(p) else p for p in params]
