from .continuous import *
from .discrete import *

all_continuous = [
    Beta,
    BetaScaled,
    Cauchy,
    Exponential,
    Gamma,
    HalfNormal,
    HalfStudent,
    Laplace,
    LogNormal,
    Normal,
    SkewNormal,
    Student,
    Uniform,
    Wald,
    Weibull,
]
all_discrete = [Binomial, DiscreteUniform, NegativeBinomial, Poisson]


__all__ = [s.__name__ for s in all_continuous] + [s.__name__ for s in all_discrete]
