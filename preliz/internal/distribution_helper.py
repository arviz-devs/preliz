import numpy as np
from scipy.special import gamma


def hdi_from_pdf(dist, mass=0.95):
    """
    Approximate the HDI by evaluating the pdf.
    This is faster, but potentially less accurate, than directly minimizing the
    interval as evaluating the ppf can be slow, specially for some distributions.
    """
    if dist.kind == "continuous":
        lower_ep, upper_ep = dist._finite_endpoints("full")
        x_vals = np.linspace(lower_ep, upper_ep, 10000)
        pdf = dist.rv_frozen.pdf(x_vals)
        pdf = pdf[np.isfinite(pdf)]
        pdf = pdf / pdf.sum()
    else:
        x_vals = dist.xvals(support="full")
        pdf = dist.rv_frozen.pmf(x_vals)

    sorted_idx = np.argsort(pdf)[::-1]
    mass_cum = 0
    indices = []
    for idx in sorted_idx:
        mass_cum += pdf[idx]
        indices.append(idx)
        if mass_cum >= mass:
            break
    return x_vals[np.sort(indices)[[0, -1]]]


def garcia_approximation(mean, sigma):
    """
    Approximate method of moments for Weibull distribution, provides good results for values of
    alpha larger than 0.83.

    Oscar Garcia. Simplified method-of-moments estimation for the Weibull distribution. 1981.
    New Zealand Journal of Forestry Science 11:304-306
    https://www.scionresearch.com/__data/assets/pdf_file/0010/36874/NZJFS1131981GARCIA304-306.pdf
    """
    ks = [-0.221016417, 0.010060668, 0.117358987, -0.050999126]  # pylint: disable=invalid-name
    z = sigma / mean  # pylint: disable=invalid-name

    poly = 0
    for idx, k in enumerate(ks):
        poly += k * z**idx

    alpha = 1 / (z * (1 + (1 - z) ** 2 * poly))
    beta = 1 / (gamma(1 + 1 / alpha) / (mean))
    return alpha, beta


def all_not_none(*args):
    for arg in args:
        if arg is None:
            return False
    return True


def any_not_none(*args):
    for arg in args:
        if arg is not None:
            return True
    return False


def valid_scalar_params(self, check_frozen=True):
    if not self.is_frozen:
        if check_frozen:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )
        return False

    if self.kind not in ["discrete", "continuous"]:
        return True

    if (
        all(isinstance(param, (int, float, np.int64)) for param in self.params)
        or self.__class__.__name__ == "Categorical"
    ):
        return True

    raise ValueError("parameters must be integers or floats")


def valid_distribution(self):
    if self.__class__.__name__ != "Categorical":
        return True

    raise ValueError(f"{self.__class__.__name__} is not supported")


init_vals = {
    "AsymmetricLaplace": {"kappa": 1.0, "mu": 0.0, "b": 1.0},
    "Beta": {"alpha": 2, "beta": 2},
    "BetaScaled": {"alpha": 2, "beta": 2, "lower": -1.0, "upper": 2.0},
    "Cauchy": {"alpha": 0.0, "beta": 1.0},
    "ChiSquared": {"nu": 5.0},
    "ExGaussian": {"mu": 0.0, "sigma": 1, "nu": 2},
    "Exponential": {"lam": 1},
    "Gamma": {"alpha": 2.0, "beta": 5.0},
    "Gumbel": {"mu": 2.0, "beta": 5.0},
    "HalfCauchy": {"beta": 1.0},
    "HalfNormal": {"sigma": 1.0},
    "HalfStudentT": {"nu": 7.0, "sigma": 1.0},
    "InverseGamma": {"alpha": 3.0, "beta": 5.0},
    "Kumaraswamy": {"a": 2.0, "b": 2.0},
    "Laplace": {"mu": 0.0, "b": 1.0},
    "Logistic": {"mu": 0.0, "s": 1},
    "LogNormal": {"mu": 0.0, "sigma": 0.5},
    "LogitNormal": {"mu": 0.0, "sigma": 0.5},
    "Moyal": {"mu": 0.0, "sigma": 1.0},
    "Normal": {"mu": 0.0, "sigma": 1.0},
    "Pareto": {"alpha": 5, "m": 2.0},
    "Rice": {"nu": 2.0, "sigma": 1.0},
    "SkewNormal": {"mu": 0.0, "sigma": 1, "alpha": 6.0},
    "StudentT": {"nu": 7, "mu": 0.0, "sigma": 1},
    "Triangular": {"lower": -2, "c": 0.0, "upper": 2.0},
    "TruncatedNormal": {"mu": 0.0, "sigma": 1, "lower": -2, "upper": 3.0},
    "Uniform": {"lower": 0.0, "upper": 1.0},
    "VonMises": {"mu": 0.0, "kappa": 1.0},
    "Wald": {"mu": 1, "lam": 3.0},
    "Weibull": {"alpha": 5.0, "beta": 2.0},
    "Bernoulli": {"p": 0.5},
    "BetaBinomial": {"alpha": 2, "beta": 2, "n": 10},
    "Binomial": {"n": 5, "p": 0.5},
    "Categorical": {"p": [0.5, 0.1, 0.4]},
    "DiscreteUniform": {"lower": -2.0, "upper": 2.0},
    "DiscreteWeibull": {"q": 0.9, "beta": 1.3},
    "Geometric": {"p": 0.5},
    "HyperGeometric": {"N": 50, "k": 10, "n": 20},
    "NegativeBinomial": {"mu": 5.0, "alpha": 2.0},
    "Poisson": {"mu": 4.5},
    "ZeroInflatedBinomial": {"psi": 0.7, "n": 10, "p": 0.5},
    "ZeroInflatedNegativeBinomial": {"psi": 0.7, "mu": 5, "alpha": 8},
    "ZeroInflatedPoisson": {"psi": 0.8, "mu": 4.5},
}
