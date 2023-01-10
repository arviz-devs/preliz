import re
from sys import modules

from preliz import distributions
from .distribution_helper import init_vals


def parse_function(source, signature):
    model = {}

    all_distributions = modules["preliz.distributions"].__all__

    all_dist_str = "|".join(all_distributions)

    source = re.sub(r"#.*$|^#.*$", "", source, flags=re.MULTILINE)
    regex = rf"(.*?({all_dist_str}).*?)\(([^()]*(?:\([^()]*\)[^()]*)*)\)"
    matches = re.finditer(regex, source)
    slidify = list(signature.parameters.keys())
    regex_coco = r"\b" + r"\b|\b".join(slidify) + r"\b"
    for match in matches:
        dist_name_str = match.group(2)
        arguments = [s.strip() for s in match.group(3).split(",")]
        args = parse_arguments(arguments, regex_coco)
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
