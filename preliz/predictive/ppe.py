import logging

from ..ppls.pymc import (
    get_model_information,
    get_guess,
    compile_logp,
    backfitting,
    write_pymc_string,
)
from ..internal.optimization import optimize_pymc_model

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
        A dictionary with the "prior" distribution of the model.
        This is actually the "projected posterior". But we call it "prior" because it is the
        prior from the perspective of the user.
    pymc_string : a string
        This is the PyMC model string with the new priors. This is what the user will want to use.
        we obtain this by taking the "prior" (the first  output) and fit it to the model's prior families
        using MLE.
    """

    _log.info(""""This is an experimental method under development, use with caution.""")

    # We collect some useful information from the model
    # probably there is a lot to improve here, to make it more robust, clean and
    # flexible
    bounds, prior, p_model, var_info, var_info2, draws = get_model_information(model)
    # initial guess for optimization routine. This is taken from model.initial_point
    # inside the optimziation routine this is updated to be estimate from the previous
    # step.
    guess = get_guess(model)
    # The log_likelihood function. This is the function we want to optimize
    # We can condition it on parameters or data
    # This will not work for prior that depends on other prior,
    # the function is "insensitive" to the top prior.
    fmodel = compile_logp(model)

    # we optimize the model to fit the target distribution
    # we actually fit to 500 samples from the target
    # so, even when we are optimizing we obtain a distribution of parameters
    # The parameters of the minimize function are based on what we do in kulprit
    # but we can tweak them to make it more robust and/or faster.
    prior = optimize_pymc_model(fmodel, target, draws, prior, guess, bounds, var_info)
    # we fit the distributions of parameters into the original families
    # in the future we could try to fit to other families
    # and return two model one with the original families and
    # one with suggested new famlies.
    new_priors = backfitting(prior, p_model, var_info2)

    return prior, write_pymc_string(new_priors, var_info2)
