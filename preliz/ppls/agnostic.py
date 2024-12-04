"""Functions to communicate with PPLs."""


from contextlib import contextmanager
import inspect
import logging
import re
import warnings


import matplotlib.pyplot as plt
import numpy as np


from preliz import distributions
from preliz.internal.distribution_helper import init_vals
from preliz.internal.plot_helper import plot_repr
from preliz.distributions import Gamma, Normal, HalfNormal
from preliz.unidimensional.mle import mle
from preliz.ppls.pymc_io import (
    extract_preliz_distributions,
    retrieve_variable_info,
    write_pymc_string,
)
from preliz.ppls.bambi_io import (
    get_pymc_model,
    write_bambi_string,
    dist_as_str,
    match_return_variables,
    dict_model,
)

try:
    from pymc import sample_prior_predictive
except ImportError:
    pass


def posterior_to_prior(model, idata, new_families=None, engine="auto"):
    """
    Fit a posterior from a model to its prior

    The fit is based on maximum likelihood of each posterior marginal to the prior
    in the model. Thus possible correlations between parameters in the posteriors
    will not be preserved. It is expected that the posterior was computed from the model.

    Parameters
    ----------
    model : A PyMC or a Bambi Model
    idata : InferenceData
        InferenceData with a posterior group.
    new_families : "auto", list or dict
        Defaults to None, the samples are fit to the original prior distribution.
        If "auto", the method evaluates the fit to the original prior plus a set of
        predefined distributions.
        Use a list of PreliZ distribution to specify the alternative distributions
        you want to consider.
        Use a dict with variables names in ``model`` as keys and a list of PreliZ
        distributions as values. This allows to specify alternative distributions
        per variable.
    engine : "auto", "pymc" or "bambi"
        Library used to define the model. Either `pymc` or `bambi`. Default is `auto`.
        The function will automatically select the appropriate library to use based on the model
        provided.
    """
    warnings.warn(""""This is an experimental method under development, use with caution.""")
    engine = get_engine(model) if engine == "auto" else engine

    if engine == "bambi":
        model = get_pymc_model(model)

    preliz_model = extract_preliz_distributions(model)
    var_info, _ = retrieve_variable_info(model)

    new_priors = back_fitting_idata(idata, preliz_model, new_families)

    if engine == "bambi":
        new_model = write_bambi_string(new_priors, var_info)
    elif engine == "pymc":
        new_model = write_pymc_string(new_priors, var_info)

    return new_model


def back_fitting_idata(idata, model_info, new_families):
    new_priors = {}
    posterior = idata.posterior.stack(sample=("chain", "draw"))

    if new_families is None:
        for var, dist in model_info.items():
            idx, _ = mle([dist], posterior[var].values, plot=False)
            new_priors[var] = dist
    else:
        for var, dist in model_info.items():
            dists = [dist]

            if new_families == "auto":
                alt = [Normal(), HalfNormal(), Gamma()]
                dists += [a for a in alt if dist.__class__.__name__ != a.__class__.__name__]
            elif isinstance(new_families, list):
                dists += new_families
            elif isinstance(new_families, dict):
                dists += new_families.get(var, [])

            idx, _ = mle(dists, posterior[var].values, plot=False)
            new_priors[var] = dists[idx[0]]
    return new_priors


def inspect_source(fmodel):
    source = inspect.getsource(fmodel)
    signature = inspect.signature(fmodel)
    source = re.sub(r"#.*$|^#.*$", "", source, flags=re.MULTILINE)
    default_params = {
        name: (param.default if param.default is not inspect.Parameter.empty else np.nan)
        for name, param in signature.parameters.items()
    }
    model = fmodel(**default_params)
    return source, signature, get_engine(model)


def get_engine(model):
    if getattr(model, "basic_RVs", False):
        return "pymc"
    elif getattr(model, "formula", False):
        return "bambi"
    return "preliz"


def parse_function_for_pred_textboxes(source, signature, engine="preliz"):
    model = {}

    slidify = list(signature.parameters.keys())
    regex = r"\b" + r"\b|\b".join(slidify) + r"\b"

    all_dist_str = dist_as_str()
    matches = match_preliz_dist(all_dist_str, source, engine)

    for match in matches:
        if engine == "bambi":
            dist_name_str = match.group(1)
        else:
            dist_name_str = match.group(2)

        if engine == "bambi":
            arguments = [s.strip() for s in match.group(2).split(",")]
        else:
            arguments = [s.strip() for s in match.group(3).split(",")]

        args = parse_arguments(arguments, regex, engine)
        for arg in args:
            if arg:
                func, var, idx = arg
                dist = getattr(distributions, dist_name_str)
                model[var] = (dist(**init_vals[dist_name_str]), idx, func)

    return model


def parse_arguments(lst, regex, engine):
    result = []
    if engine == "pymc":
        offset = 1
    else:
        offset = 0
    for idx, item in enumerate(lst):
        match = re.search(regex, item)
        if match:
            if item.isidentifier():
                result.append((None, match.group(0), idx - offset))
            else:
                if "**" in item:
                    power = item.split("**")[1].strip()
                    result.append((power, match.group(0), idx - offset))
                else:
                    func = item.split("(")[0].split(".")[-1]
                    result.append((func, match.group(0), idx - offset))
    return result


def get_prior_pp_samples(fmodel, variables, draws, engine=None, values=None):
    if values is None:
        values = []

    if engine == "preliz":
        obs_rv = variables[-1]  # only one observed for the moment
        pp_samples_ = []
        prior_samples_ = {name: [] for name in variables[:-1]}
        for _ in range(draws):
            for name, value in zip(variables, fmodel(*values)):
                if name == obs_rv:
                    pp_samples_.append(value)
                else:
                    prior_samples_[name].append(value)

        pp_samples = np.stack(pp_samples_)
        prior_samples = {key: np.array(val) for key, val in prior_samples_.items()}
    elif engine == "bambi":
        *prior_samples_, pp_samples = fmodel(*values)
        prior_samples = {name: np.array(val) for name, val in zip(variables[:-1], prior_samples_)}

    return pp_samples, prior_samples


def from_preliz(fmodel):
    source = inspect.getsource(fmodel)
    variables = match_return_variables(source)
    # Find the priors we want to change
    all_dist_str = dist_as_str()
    matches = match_preliz_dist(all_dist_str, source, "preliz")
    # Create a dictionary with the priors
    model = dict_model(matches, variables)

    return variables, model


def match_preliz_dist(all_dist_str, source, engine):
    # remove comments
    source = re.sub(r"#.*$|^#.*$", "", source, flags=re.MULTILINE)

    if engine in ["preliz", "pymc"]:
        regex = rf"(.*?({all_dist_str}).*?)\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
    if engine == "bambi":
        regex = rf'\s*(?:\w+\.)?Prior\("({all_dist_str})",\s*((?:\w+=\w+(?:,?\s*)?)*)\s*\)'
    matches = re.finditer(regex, source)
    return matches


def ppl_plot_decorator(func, iterations, kind_plot, references, plot_func, engine):
    def looper(*args, **kwargs):
        kwargs.pop("__resample__")
        x_min = kwargs.pop("__x_min__")
        x_max = kwargs.pop("__x_max__")
        if not kwargs.pop("__set_xlim__"):
            x_min = None
            x_max = None
            auto = True
        else:
            auto = False

        if engine == "preliz":
            results = []
            for _ in range(iterations):
                val = func(*args, **kwargs)
                if not any(np.isnan(val)):
                    results.append(val)
            results = np.array(results)

        elif engine == "bambi":
            model = func(*args, **kwargs)
            model.build()
            with disable_pymc_sampling_logs():
                idata = model.prior_predictive(iterations)
            results = (
                idata["prior_predictive"]
                .stack(sample=("chain", "draw"))[model.response_component.response.name]
                .values.T
            )

        elif engine == "pymc":
            with func(*args, **kwargs) as model:
                obs_name = model.observed_RVs[0].name
                with disable_pymc_sampling_logs():
                    idata = sample_prior_predictive(samples=iterations)
                results = (
                    idata["prior_predictive"].stack(sample=("chain", "draw"))[obs_name].values.T
                )

        _, ax = plt.subplots()
        ax.set_xlim(x_min, x_max, auto=auto)
        if plot_func is None:
            plot_repr(results, kind_plot, references, iterations, ax)
        else:
            plot_func(results, ax)

    return looper


@contextmanager
def disable_pymc_sampling_logs(logger: logging.Logger = logging.getLogger("pymc")):
    effective_level = logger.getEffectiveLevel()
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(effective_level)
