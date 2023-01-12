import inspect
import re
from sys import modules

import numpy as np

from preliz import distributions
from .distribution_helper import init_vals


def inspect_source(fmodel):
    source = inspect.getsource(fmodel)
    signature = inspect.signature(fmodel)
    source = re.sub(r"#.*$|^#.*$", "", source, flags=re.MULTILINE)

    return source, signature


def parse_function_for_pred_sliders(source, signature):
    model = {}

    slidify = list(signature.parameters.keys())
    regex = r"\b" + r"\b|\b".join(slidify) + r"\b"

    matches = match_preliz_dist(source)

    for match in matches:
        dist_name_str = match.group(2)
        arguments = [s.strip() for s in match.group(3).split(",")]
        args = parse_arguments(arguments, regex)
        for arg in args:
            if arg:
                func, var, idx = arg
                dist = getattr(distributions, dist_name_str)
                model[var] = (dist(**init_vals[dist_name_str]), idx, func)

    return model


def parse_arguments(lst, regex):
    result = []
    for idx, item in enumerate(lst):
        match = re.search(regex, item)
        if match:
            if item.isidentifier():
                result.append((None, match.group(0), idx))
            else:
                if "**" in item:
                    power = item.split("**")[1].strip()
                    result.append((power, match.group(0), idx))
                else:
                    func = item.split("(")[0].split(".")[-1]
                    result.append((func, match.group(0), idx))
    return result


def get_prior_pp_samples(fmodel, draws):
    match = match_return_variables(fmodel)
    if match:
        variables = [var.strip() for var in match.group(1).split(",")]

    obs_rv = variables[-1]  # only one observed for the moment
    pp_samples_ = []
    prior_samples_ = {name: [] for name in variables[:-1]}
    for _ in range(draws):
        for name, value in zip(variables, fmodel()):
            if name == obs_rv:
                pp_samples_.append(value)
            else:
                prior_samples_[name].append(value)

    pp_samples = np.stack(pp_samples_)
    prior_samples = {key: np.array(val) for key, val in prior_samples_.items()}

    return pp_samples, prior_samples, obs_rv


def parse_function_for_ppa(source, obs_rv):
    model = {}

    matches = match_preliz_dist(source)
    for match in matches:
        var_name = match.group(0).split("=")[0].strip()
        if var_name != obs_rv:
            dist = getattr(modules["preliz.distributions"], match.group(2))
            model[var_name] = dist()

    return model


def match_preliz_dist(source):
    all_distributions = modules["preliz.distributions"].__all__
    all_dist_str = "|".join(all_distributions)

    regex = rf"(.*?({all_dist_str}).*?)\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
    matches = re.finditer(regex, source)
    return matches


def match_return_variables(fmodel):
    source = inspect.getsource(fmodel)
    match = re.search(r"return (\w+(\s*,\s*\w+)*)", source)
    return match
