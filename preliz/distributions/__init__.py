from .continuous import *
from .discrete import *

all_continuous = [
    Beta,
    BetaScaled,
    Cauchy,
    Exponential,
    Gamma,
    HalfCauchy,
    HalfNormal,
    HalfStudent,
    InverseGamma,
    Laplace,
    LogNormal,
    Normal,
    Pareto,
    SkewNormal,
    Student,
    TruncatedNormal,
    Uniform,
    Wald,
    Weibull,
]
all_discrete = [Binomial, DiscreteUniform, NegativeBinomial, Poisson]


__all__ = [s.__name__ for s in all_continuous] + [s.__name__ for s in all_discrete]
