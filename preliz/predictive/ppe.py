from ..ppls.pymc import (
    get_model_information,
    get_guess,
    compile_logp,
    backfitting,
    write_pymc_string,
)
from ..internal.optimization import optimize_pymc_model


def ppe(model, target):
    """
    Projective Predictive Elicitation
    """
    bounds, prior, p_model, var_info, var_info2, draws = get_model_information(model)
    guess = get_guess(model)
    fmodel = compile_logp(model)

    prior = optimize_pymc_model(fmodel, target, draws, prior, guess, bounds, var_info)
    new_priors = backfitting(prior, p_model, var_info2)

    return prior, write_pymc_string(new_priors, var_info2)
