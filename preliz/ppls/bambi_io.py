"""Functions to communicate with Bambi."""

import importlib
import inspect
from copy import copy
import re
from sys import modules

import numpy as np


def get_pymc_model(model):
    if not model.built:
        model.build()
    pymc_model = model.backend.model
    return pymc_model


def write_bambi_string(new_priors, var_info):
    """
    Return a string with the new priors for the Bambi model.
    So the user can copy and paste, ideally with none to minimal changes.
    """
    header = "{\n"
    variables = []
    names = list(new_priors.keys())
    for key, value in new_priors.items():
        idxs = var_info[key][-1]
        if idxs:
            for i in idxs:
                nkey = names[i]
                cp_dist = copy(new_priors[nkey])
                cp_dist._fit_moments(np.mean(value.mean()), np.mean(value.std()))

                dist_name, dist_params = repr(cp_dist).split("(")
                size = var_info[nkey][1]
                if size > 1:
                    variables[
                        i
                    ] = f'"{nkey}" : bmb.Prior("{dist_name}", {dist_params}, shape={size}),\n'
                else:
                    variables[i] = f'"{nkey}" : bmb.Prior("{dist_name}", {dist_params}),\n'
        else:
            dist_name, dist_params = repr(value).split("(")
            dist_params = dist_params.rstrip(")")
            size = var_info[key][1]
            if size > 1:
                variables.append(
                    f'"{key}" : bmb.Prior("{dist_name}", {dist_params}, shape={size}),\n'
                )
            else:
                variables.append(f'"{key}" : bmb.Prior("{dist_name}", {dist_params}),\n')

    return "".join([header] + variables)


def from_bambi(fmodel, draws):
    module_name = fmodel.__module__
    module = importlib.import_module(module_name)

    # Get the source code of the original function
    original_source = inspect.getsource(fmodel)

    # Define a pattern to find the line where the model is built
    pattern = re.compile(r"(\s+)([a-zA-Z_]\w*)\s*=\s*.*?Model(.*)")

    # Find the match in the source code
    match = pattern.search(original_source)

    # Extract the indentation and variable name
    indentation = match.group(1)
    variable_name = match.group(2)

    # Find the variables after the return statement
    return_variables = match_return_variables(original_source)

    if return_variables:
        # Build the new source code
        new_source = original_source.replace(
            match.group(0),
            f"{match.group(0)}"
            f"{indentation}{variable_name}.build()\n"
            f"{indentation}variables = [{variable_name}.backend.model.named_vars[v] "
            f"for v in {return_variables}]\n"
            f'{indentation}{", ".join(return_variables)} = pm.draw(variables, draws={draws})',
        )

        # Find the priors we want to change
        all_dist_str = dist_as_str()
        matches = match_preliz_dist(all_dist_str, new_source)
        # Create a dictionary with the priors
        model = dict_model(matches, return_variables)

        # Execute the new source code to redefine the function
        exec(new_source, module.__dict__)  # pylint: disable=exec-used
        modified_fmodel = getattr(module, fmodel.__name__)

    return modified_fmodel, return_variables, model


def match_preliz_dist(all_dist_str, source):
    # remove comments
    source = re.sub(r"#.*$|^#.*$", "", source, flags=re.MULTILINE)
    regex = rf'\s*(?:\w+\.)?Prior\("({all_dist_str})",\s*((?:\w+=\w+(?:,?\s*)?)*)\s*\)'
    matches = re.finditer(regex, source)
    return matches


def match_return_variables(source):
    match = re.search(r"return (\w+(\s*,\s*\w+)*)", source)
    return [var.strip() for var in match.group(1).split(",")]


def dist_as_str():
    all_distributions = modules["preliz.distributions"].__all__
    return "|".join(all_distributions)


def dict_model(matches, return_variables):
    model = {}
    obs_rv = return_variables[-1]
    for match in matches:
        var_name = match.group(0).split("=")[0].strip()
        if var_name != obs_rv:
            dist = getattr(modules["preliz.distributions"], match.group(2))
            model[var_name] = dist()

    return model
