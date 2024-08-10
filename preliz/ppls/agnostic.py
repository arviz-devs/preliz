"""Functions to communicate with PPLs."""

from preliz.distributions import Gamma, Normal, HalfNormal
from preliz.unidimensional.mle import mle
from preliz.ppls.pymc_io import get_model_information


def posterior_to_prior(model, posterior, alternative=None):
    """
    Fit a posterior from a model to its prior

    The fit is based on maximum likelihood of each posterior marginal to the prior
    in the model. It is expected that the posterior was computed from the model.

    Parameters
    ----------
    model : A PyMC model
    posterior : InferenceData
        InferenceData with a posterior group.
    alternative : "auto", list or dict
        Defaults to None, the samples are fit to the original prior distribution.
        If "auto", the method evaluates the fit to the original prior plus a set of
        predefined distributions.
        Use a list of PreliZ distribution to specify the alternative distributions
        you want to consider.
        Use a dict with variables names in ``model`` as key and a list of preliz
        distributions as value. This allows to specify alternative distributions
        per variable.
    """
    model_info = get_model_information(model)[2]
    new_priors = []

    if alternative is None:
        for var, dist in model_info.items():
            dist._fit_mle(posterior[var].values)
            new_priors.append((dist, var))
    else:
        for var, dist in model_info.items():
            dists = [dist]

            if alternative == "auto":
                dists += [Normal(), HalfNormal(), Gamma()]
            elif isinstance(alternative, list):
                dists += alternative
            elif isinstance(alternative, dict):
                dists += alternative.get(var, [])

            idx = mle(dists, posterior[var].values, plot=False)[0]
            new_priors.append((dists[idx[0]], var))

    new_model = "\n".join(f"{var} = {new_prior}" for new_prior, var in new_priors)
    return new_model
