import re
from sys import modules

from preliz import distributions
from .distribution_helper import init_vals


def parse_function(source):
    seen_distributions = {}
    model = {}

    all_distributions = modules["preliz.distributions"].__all__

    all_dist_str = "|".join(all_distributions)

    regex = rf"(.*?({all_dist_str}).*?)\((.*?)\)"
    matches = re.finditer(regex, source)

    for match in matches:
        var_name = match.group(0).split("=")[0].strip()
        dist_name_str = match.group(2)
        args = [s.strip() for s in match.group(3).split(",")]
        args_ = []
        for idx, arg in enumerate(args):
            arg = arg.strip()
            if arg.isnumeric():
                args_.append(float(arg))
            else:
                if "=" in arg:
                    arg = arg.split("=")[1]
                if arg in seen_distributions:
                    args_.append(seen_distributions[arg].rvs())
                else:
                    args_.append(list(init_vals[dist_name_str].values())[idx])

        dist = getattr(distributions, dist_name_str)
        seen_distributions[var_name] = dist(*args_)
        for idx, arg in enumerate(args):
            arg = arg.strip()
            if "=" in arg:
                arg = arg.split("=")[1]
            model[arg] = (dist(*args_), idx)

    return model
