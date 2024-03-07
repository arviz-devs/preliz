"""Projective predictive elicitation."""

import logging

from preliz.internal.optimization import optimize_pymc_model
from preliz.ppls.pymc_io import (
    get_model_information,
    get_guess,
    compile_logp,
    backfitting,
    write_pymc_string,
)


_log = logging.getLogger("preliz")


def ppe(model, target):
    """
    Projective Predictive Elicitation.

    This is an experimental method under development, use with caution.
    Most likely thing will break.

    Parameters
    ----------
    model : a probabilistic model
        Currently it only works with PyMC model. More PPls coming soon.
    target : a Preliz distribution
        This represent the prior predictive distribution **previously** elicited from the user,
        possibly using other Preliz's methods to obtain this distribution, such as maxent,
        roulette, quartile, etc.
        This should represent the domain-knowledge of the user and not any observed dataset.
        Currently only works with a Preliz distributions. In the future we should support mixture of
        distributions (mixture of "experts"), and maybe other options.

    Returns
    -------
    prior : a dictionary
        Prior samples approximating the prior distribution that will induce
        a prior predictive distribution close to ``target``.
    pymc_string : a string
        This is the PyMC model string with the new priors.
        Computed by taking the "prior samples" and fit it into the model's prior families
        using MLE.
    """
    _log.info(""""This is an experimental method under development, use with caution.""")

    # Get information from PyMC model
    bounds, prior, p_model, var_info, var_info2, draws, free_rvs = get_model_information(model)
    # Initial point for optimization
    guess = get_guess(model, free_rvs)
    # compile PyMC model
    fmodel = compile_logp(model)
    # find prior that induce a prior predictive distribution close to target
    prior = optimize_pymc_model(fmodel, target, draws, prior, guess, bounds, var_info, p_model)
    # Fit the prior into the model's prior
    # So we can write it as a PyMC model
    new_priors = backfitting(prior, p_model, var_info2)

    return prior, write_pymc_string(new_priors, var_info2)
