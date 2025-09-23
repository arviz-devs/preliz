import re
from functools import wraps
from sys import modules

import numpy as np
from pytensor import function
from pytensor.compile.function.types import Function, In
from pytensor.tensor import tensor
from pytensor.tensor.random.type import random_generator_type

eps = np.finfo(float).eps


def from_precision(precision):
    sigma = 1 / precision**0.5
    return sigma


def to_precision(sigma):
    precision = 1 / (eps + sigma**2)
    return precision


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


def valid_scalar_params(self, check_frozen=True, raise_error=True):
    if not self.is_frozen:
        if check_frozen:
            raise ValueError(
                "Undefined distribution, "
                "you need to first define its parameters or use one of the fit methods"
            )
        return False

    if self.kind not in ["discrete", "continuous"]:
        return True

    if all(
        isinstance(param, int | float | np.int64) for param in self.params
    ) or self.__class__.__name__ in ["Categorical", "Mixture"]:
        return True

    if raise_error:
        raise ValueError("parameters must be integers or floats")

    return False


def valid_distribution(self):
    if self.__class__.__name__ != "Categorical":
        return True

    raise ValueError(f"{self.__class__.__name__} is not supported")


def process_extra(input_string):
    pattern = r"(\w+)\((.*?)\)"
    matches = re.findall(pattern, input_string)
    result_dict = {}
    for match in matches:
        name = match[0]
        args = match[1].split(",")
        arg_dict = {}
        try:
            for arg in args:
                key, value = arg.split("=")
                arg_dict[key.strip()] = float(value)
            result_dict[name] = arg_dict
        except ValueError:
            pass
    return result_dict


def num_skewness(dist):
    mean = dist.mean()
    std = dist.std()
    x_values = dist.xvals("full")
    pdf = dist.pdf(x_values)
    if dist.kind == "discrete":
        return np.sum(((x_values - mean) / std) ** 3 * pdf)
    else:
        return np.trapezoid(((x_values - mean) / std) ** 3 * pdf, x_values)


def num_kurtosis(dist):
    mean = dist.mean()
    std = dist.std()
    x_values = dist.xvals("full")
    pdf = dist.pdf(x_values)
    if dist.kind == "discrete":
        return np.sum(((x_values - mean) / std) ** 4 * pdf) - 3
    else:
        return np.trapezoid(((x_values - mean) / std) ** 4 * pdf, x_values) - 3


def pytensor_jit(func, **compile_kwargs):
    # trust_input can be a problem if the user passes aliased inputs
    # (including the same values twice)
    # But it really reduces the overhead!
    # Removing inplace rewrites would make this safe always (I think)
    compile_kwargs.setdefault("trust_input", True)
    compile_kwargs.setdefault("mode", "NUMBA")
    compile_kwargs.setdefault("on_unused_input", "ignore")
    signature_to_function: tuple[int, Function] = {}

    @wraps(func)
    def inner_func(*args, signature_to_function=signature_to_function):
        args = [np.asarray(a) for a in args]
        signature = tuple((tuple(s == 1 for s in a.shape), a.dtype) for a in args)

        try:
            return signature_to_function[signature](*args)
        except KeyError:
            pass
        symbolic_args = [
            tensor(broadcastable=bcast_pattern, dtype=dtype) for (bcast_pattern, dtype) in signature
        ]
        symbolic_out = func(*symbolic_args)
        signature_to_function[signature] = compiled_func = function(
            symbolic_args, symbolic_out, **compile_kwargs
        )
        return compiled_func(*args)

    return inner_func


def pytensor_rng_jit(_func=None, **compile_kwargs):
    """Compile pytensor function with RNG on demand."""

    def decorator(func):
        compile_kwargs.setdefault("trust_input", True)
        compile_kwargs.setdefault("mode", "NUMBA")

        signature_to_function: dict[tuple, Function] = {}

        @wraps(func)
        def inner_func(*args, size, rng, signature_to_function=signature_to_function):
            args = tuple(np.asarray(a) for a in args)

            try:
                if size is None:
                    signature = (
                        *((tuple(s == 1 for s in a.shape), a.dtype) for a in args),
                        None,
                    )
                    return signature_to_function[signature](*args, rng)
                else:
                    if isinstance(size, (int | np.integer)):
                        size = (size,)
                    size = np.asarray(size, dtype="int64")
                    signature = (
                        *((tuple(s == 1 for s in a.shape), a.dtype) for a in args),
                        tuple(size),
                    )
                    return signature_to_function[signature](*args, size, rng)
            except KeyError:
                pass

            symbolic_args = [
                tensor(broadcastable=bcast_pattern, dtype=dtype)
                for (bcast_pattern, dtype) in signature[:-1]
            ]
            symbolic_size = (
                None if size is None else tensor(shape=(len(size),), dtype="int64", name="size")
            )
            symbolic_rng = random_generator_type("rng")
            symbolic_out = func(*symbolic_args, size=symbolic_size, rng=symbolic_rng)

            # We allow PyTensor to modify the RNG
            mutable_rng = In(symbolic_rng, mutable=True)
            if size is None:
                symbolic_inputs = [*symbolic_args, mutable_rng]
            else:
                symbolic_inputs = [*symbolic_args, symbolic_size, mutable_rng]

            signature_to_function[signature] = compiled_func = function(
                symbolic_inputs,
                symbolic_out,
                **compile_kwargs,
            )
            return compiled_func(*args, rng) if size is None else compiled_func(*args, size, rng)

        return inner_func

    if _func is None:
        return decorator
    else:
        return decorator(_func)


init_vals = {
    "AsymmetricLaplace": {"kappa": 1.3, "mu": 0.0, "b": 1.0},
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
    "LogLogistic": {"alpha": 1, "beta": 5},
    "Logistic": {"mu": 0.0, "s": 1},
    "LogNormal": {"mu": 0.0, "sigma": 0.5},
    "LogitNormal": {"mu": 0.0, "sigma": 0.5},
    "Moyal": {"mu": 0.0, "sigma": 1.0},
    "Normal": {"mu": 0.0, "sigma": 1.0},
    "Pareto": {"alpha": 5, "m": 2.0},
    "Rice": {"nu": 2.0, "sigma": 1.0},
    "ScaledInverseChiSquared": {"nu": 10, "tau2": 1.0},
    "SkewNormal": {"mu": 0.0, "sigma": 1, "alpha": 6.0},
    "SkewStudentT": {"mu": 0.0, "sigma": 1, "a": 2, "b": 2},
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
    "Truncated": {"lower": -10, "upper": 10},
    "Censored": {"lower": -10, "upper": 10},
    "Dirichlet": {"alpha": [1.0, 1.0, 1.0]},
}


def get_distributions(dist_names=None):
    if dist_names is None:
        all_distributions = modules["preliz.distributions"].__all__
    else:
        all_distributions = dist_names

    distributions = []
    for a_dist in all_distributions:
        dist = getattr(modules["preliz.distributions"], a_dist)()
        distributions.append(dist)
    return distributions
