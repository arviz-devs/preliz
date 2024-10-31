"""Projective predictive elicitation."""

import warnings
import numpy as np

from preliz.internal.optimization import optimize_pymc_model
from preliz.ppls.bambi_io import get_pymc_model, write_bambi_string
from preliz.ppls.agnostic import back_fitting_idata
from preliz.internal.parser import get_engine
from preliz.ppls.pymc_io import (
    get_model_information,
    get_initial_guess,
    compile_logp,
    back_fitting_pymc,
    write_pymc_string,
)


def ppe(model, target, method="projective", engine="auto", random_state=0):
    """
    Prior Predictive Elicitation.

    This method is experimental and under development. It does not offers guarantees of
    correctness. Use with caution and triple-check the results.

    Parameters
    ----------
    model : a probabilistic model
        Currently it only works with PyMC model. More PPls coming soon.
    method : str
        Method used to generate samples that match the target distribution.
        Defaults to `"projective"`, another option is `"pathfinder"`.
        If `"projective"`, the parameters of the priors are only used to provide an initial
        guess for the optimization routine. Thus their effect on the result is smaller than in
        traditional Bayesian inference, unless the priors are very vague or very strong.
        If `"projective"`, the observed values are ignored, but not their size.
        Pathfinder is a variational inference method so the role of the priors and observed values
        is what is expected in Bayesian inference.
    engine : str
        Library used to define the model. Either `"auto"` (default), `"pymc"` or `"bambi"`.
        Ig `"auto"`, the library is automatically detected.
    target : a PreliZ distribution or list
        Instance of a PreliZ distribution or a list of tuples where each tuple contains a PreliZ
        distribution and a weight.
        This represents the prior predictive distribution **previously** elicited by the user,
        possibly using other PreliZ's methods to obtain this distribution, such as maxent,
        roulette, quartile, etc.
        This should represent the domain-knowledge of the user and not any observed dataset.
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

    rng = np.random.default_rng(random_state)
    engine = get_engine(model) if engine == "auto" else engine

    # Get models information
    if engine == "bambi":
        model = get_pymc_model(model)
    (
        bounds,
        prior,
        preliz_model,
        transformed_var_info,
        untransformed_var_info,
        num_draws,
        free_rvs,
    ) = get_model_information(model)

    # With the projective method we attempt to find a prior that induces
    # a prior predictive distribution as close as possible to the target distribution
    if method == "projective":
        # Initial point for optimization
        initial_guess = get_initial_guess(model, free_rvs)
        # compile PyMC model
        fmodel = compile_logp(model)
        projection = optimize_pymc_model(
            fmodel,
            target,
            num_draws,
            bounds,
            initial_guess,
            prior,
            preliz_model,
            transformed_var_info,
            rng,
        )
        # Backfit `projected_posterior` into the model's prior-families
        new_priors = back_fitting_pymc(projection, preliz_model, untransformed_var_info)
        if engine == "bambi":
            return write_bambi_string(new_priors, untransformed_var_info)
        if engine == "pymc":
            return write_pymc_string(new_priors, untransformed_var_info)

    # Fit the samples to the original prior distribution
    # or to a set of predefined distributions
    elif method == "pathfinder":
        from pymc_experimental import fit  # pylint:disable=import-outside-toplevel

        with model:
            idata = fit(method="pathfinder", num_samples=1000)

    new_priors = back_fitting_idata(idata, preliz_model, alternative=False)
    if engine == "bambi":
        new_model = write_bambi_string(new_priors, untransformed_var_info)
    elif engine == "pymc":
        new_model = write_pymc_string(new_priors, untransformed_var_info)

    return new_model
