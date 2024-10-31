"""Functions to communicate with Bambi."""


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
    for key, value in new_priors.items():
        dist_name, dist_params = repr(value).split("(")
        dist_params = dist_params.rstrip(")")
        size = var_info[key][1]
        if size > 1:
            header += f'"{key}" : bmb.Prior("{dist_name}", {dist_params}, shape={size}),\n'
        else:
            header += f'"{key}" : bmb.Prior("{dist_name}", {dist_params}),\n'

    header = header.rstrip(", ") + "}"
    return header
