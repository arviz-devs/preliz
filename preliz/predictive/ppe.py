"""Projective predictive elicitation."""

import warnings

import numpy as np

from preliz.internal.optimization import optimize_pymc_model
from preliz.ppls.agnostic import get_engine
from preliz.ppls.bambi_io import get_pymc_model, write_bambi_string
from preliz.ppls.pymc_io import (
    back_fitting_pymc,
    compile_mllk,
    extract_preliz_distributions,
    get_initial_guess,
    retrieve_variable_info,
    unravel_projection,
    write_pymc_string,
)


def ppe(model, target, engine="auto", new_families=None, random_state=0):
    """
    Prior Predictive Elicitation.

    This method is experimental and under development. It does not offers guarantees of
    correctness. Use with caution and triple-check the results.

    With the projective method we attempt to find a prior that induces
    a prior predictive distribution as close as possible to the target distribution

    Parameters
    ----------
    model : a probabilistic model
        Currently it only works with PyMC model. More PPls coming soon.
    target : a PreliZ distribution or list
        Instance of a PreliZ distribution or a list of tuples where each tuple contains a PreliZ
        distribution and a weight.
        This represents the prior predictive distribution **previously** elicited by the user,
        possibly using other PreliZ's methods to obtain this distribution, such as maxent,
        roulette, quartile, etc.
        This should represent the domain-knowledge of the user and not any observed dataset.
    engine : str
        Library used to define the model. Either `"auto"` (default), `"pymc"` or `"bambi"`.
        Ig `"auto"`, the library is automatically detected.
    new_families : "auto", list or dict
        Defaults to None, the samples are fit to the original prior distribution.
        If "auto", the method evaluates the fit to the original prior plus a set of
        predefined distributions.
        Use a list of PreliZ distribution to specify the alternative distributions
        you want to consider.
        Use a dict with variables names in ``model`` as keys and a list of PreliZ
        distributions as values. This allows to specify alternative distributions
        per variable.
    random_state : {None, int, numpy.random.Generator, numpy.random.RandomState}
        Defaults to 0. Ignored if `method` is `"pathfinder"`.

    Returns
    -------
    new_priors : str
        A string representation of the new priors. The user can copy and paste it into
        the model's code. Ideally, with none to minimal changes.
    """
    warnings.warn(
        """This method is experimental and under development with no guarantees of correctness.
                  Use with caution and triple-check the results."""
    )
    opt_iterations = 400

    rng = np.random.default_rng(random_state)
    engine = get_engine(model) if engine == "auto" else engine

    # Get models information
    if engine == "bambi":
        model = get_pymc_model(model)

    preliz_model = extract_preliz_distributions(model)
    var_info, num_draws = retrieve_variable_info(model)

    # Initial point for optimization
    initial_guess = get_initial_guess(model)
    # compile PyMC model
    fmodel, old_y_value, obs_rvs = compile_mllk(model)
    projection_raveled = optimize_pymc_model(
        fmodel,
        target,
        num_draws,
        opt_iterations,
        initial_guess,
        rng,
    )
    # restore obs_rvs value in the model
    model.rvs_to_values[obs_rvs] = old_y_value

    projection_unraveled = unravel_projection(projection_raveled, var_info, opt_iterations)

    # Backfit `projected_posterior` into the model's prior-families
    projection_backfitted = back_fitting_pymc(
        projection_unraveled, preliz_model, var_info, new_families
    )

    if engine == "bambi":
        new_priors = write_bambi_string(projection_backfitted, var_info)
    elif engine == "pymc":
        new_priors = write_pymc_string(projection_backfitted, var_info)

    return new_priors
