from preliz.ppls.pymc_io import get_model_information


def get_bmb_model_information(model):
    if not model.built:
        model.build()
    pymc_model = model.backend.model
    return get_model_information(pymc_model)


def write_bambi_string(new_priors, var_info):
    """
    Return a string with the new priors for the Bambi model.
    So the user can copy and paste, ideally with none to minimal changes.
    """
    header = "{"
    for key, value in new_priors.items():
        dist_name, dist_params = repr(value).split("(")
        dist_params = dist_params.rstrip(")")
        size = var_info[key][1]
        if size > 1:
            header += f'"{key}" : bmb.Prior("{dist_name}", {dist_params}, shape={size}), '
        else:
            header += f'"{key}" : bmb.Prior("{dist_name}", {dist_params}), '

    header = header.rstrip(", ") + "}"
    return header
