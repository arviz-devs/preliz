# Continuous Distributions
from preliz.distributions.asymmetric_laplace import AsymmetricLaplace

# Discrete Distributions
from preliz.distributions.bernoulli import Bernoulli
from preliz.distributions.beta import Beta
from preliz.distributions.betabinomial import BetaBinomial
from preliz.distributions.betascaled import BetaScaled
from preliz.distributions.binomial import Binomial
from preliz.distributions.categorical import Categorical
from preliz.distributions.cauchy import Cauchy
from preliz.distributions.censored import Censored
from preliz.distributions.chi_squared import ChiSquared

# Multivariate Distributions
from preliz.distributions.continuous_multivariate import Dirichlet, MvNormal
from preliz.distributions.discrete_uniform import DiscreteUniform
from preliz.distributions.discrete_weibull import DiscreteWeibull
from preliz.distributions.exgaussian import ExGaussian
from preliz.distributions.exponential import Exponential
from preliz.distributions.gamma import Gamma
from preliz.distributions.geometric import Geometric
from preliz.distributions.gumbel import Gumbel
from preliz.distributions.halfcauchy import HalfCauchy
from preliz.distributions.halfnormal import HalfNormal
from preliz.distributions.halfstudentt import HalfStudentT
from preliz.distributions.hurdle import Hurdle
from preliz.distributions.hypergeometric import HyperGeometric
from preliz.distributions.inversegamma import InverseGamma
from preliz.distributions.kumaraswamy import Kumaraswamy
from preliz.distributions.laplace import Laplace
from preliz.distributions.logistic import Logistic
from preliz.distributions.logitnormal import LogitNormal
from preliz.distributions.loglogistic import LogLogistic
from preliz.distributions.lognormal import LogNormal

# Transform Distributions
from preliz.distributions.mixture import Mixture
from preliz.distributions.moyal import Moyal
from preliz.distributions.negativebinomial import NegativeBinomial
from preliz.distributions.normal import Normal
from preliz.distributions.pareto import Pareto
from preliz.distributions.poisson import Poisson
from preliz.distributions.rice import Rice
from preliz.distributions.skew_studentt import SkewStudentT
from preliz.distributions.skewnormal import SkewNormal
from preliz.distributions.studentt import StudentT
from preliz.distributions.triangular import Triangular
from preliz.distributions.truncated import Truncated
from preliz.distributions.truncatednormal import TruncatedNormal
from preliz.distributions.uniform import Uniform
from preliz.distributions.vonmises import VonMises
from preliz.distributions.wald import Wald
from preliz.distributions.weibull import Weibull
from preliz.distributions.zi_binomial import ZeroInflatedBinomial
from preliz.distributions.zi_negativebinomial import ZeroInflatedNegativeBinomial
from preliz.distributions.zi_poisson import ZeroInflatedPoisson

all_continuous = [
    AsymmetricLaplace,
    Beta,
    BetaScaled,
    Cauchy,
    ChiSquared,
    ExGaussian,
    Exponential,
    Gamma,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    HalfStudentT,
    InverseGamma,
    Kumaraswamy,
    Laplace,
    Logistic,
    LogLogistic,
    LogNormal,
    LogitNormal,
    Moyal,
    Normal,
    Pareto,
    Rice,
    SkewNormal,
    SkewStudentT,
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    VonMises,
    Wald,
    Weibull,
]
all_discrete = [
    Bernoulli,
    BetaBinomial,
    Binomial,
    Categorical,
    DiscreteUniform,
    DiscreteWeibull,
    Geometric,
    HyperGeometric,
    NegativeBinomial,
    Poisson,
    ZeroInflatedBinomial,
    ZeroInflatedNegativeBinomial,
    ZeroInflatedPoisson,
]

all_continuous_multivariate = [Dirichlet, MvNormal]


__all__ = (  # noqa: PLE0604
    [s.__name__ for s in all_continuous]
    + [s.__name__ for s in all_discrete]
    + [s.__name__ for s in all_continuous_multivariate]
    + [Mixture.__name__]
    + [Truncated.__name__]
    + [Censored.__name__]
    + [Hurdle.__name__]
)
