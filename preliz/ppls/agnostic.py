"""Functions to communicate with PPLs."""

from preliz.distributions import Gamma, Normal, HalfNormal
from preliz.unidimensional.mle import mle
from preliz.ppls.pymc_io import get_model_information, write_pymc_string


def posterior_to_prior(model, idata, alternative=None):
    """
    Fit a posterior from a model to its prior

    The fit is based on maximum likelihood of each posterior marginal to the prior
    in the model. It is expected that the posterior was computed from the model.

    Parameters
    ----------
    model : A PyMC model
    idata : InferenceData
        InferenceData with a posterior group.
    alternative : "auto", list or dict
        Defaults to None, the samples are fit to the original prior distribution.
        If "auto", the method evaluates the fit to the original prior plus a set of
        predefined distributions.
        Use a list of PreliZ distribution to specify the alternative distributions
        you want to consider.
        Use a dict with variables names in ``model`` as keys and a list of PreliZ
        distributions as values. This allows to specify alternative distributions
        per variable.
    """
    _, _, model_info, _, var_info2, *_ = get_model_information(model)
    new_priors = {}
    posterior = idata.posterior.stack(sample=("chain", "draw"))

    if alternative is None:
        for var, dist in model_info.items():
            print(var)
            dist._fit_mle(posterior[var].values)
            new_priors[var] = dist
    else:
        for var, dist in model_info.items():
            dists = [dist]

            if alternative == "auto":
                dists += [Normal(), HalfNormal(), Gamma()]
            elif isinstance(alternative, list):
                dists += alternative
            elif isinstance(alternative, dict):
                dists += alternative.get(var, [])

            idx, _ = mle(dists, posterior[var].values, plot=False)
            new_priors[var] = dists[idx[0]]

    new_model = write_pymc_string(new_priors, var_info2)

    return new_model
