"""PreliZ rcparams. Based on ArviZ's implementation."""

import locale
import os
import pprint
import re
import sys
from collections.abc import MutableMapping
from pathlib import Path
import warnings

import numpy as np


def _make_validate_choice(accepted_values, allow_none=False, typeof=str):
    """Validate value is in accepted_values.

    Parameters
    ----------
    accepted_values : iterable
        Iterable containing all accepted_values.
    allow_none: boolean, optional
        Whether to accept ``None`` in addition to the values in ``accepted_values``.
    typeof: type, optional
        Type the values should be converted to.
    """
    # no blank lines allowed after function docstring by pydocstyle,
    # but black requires white line before function

    def validate_choice(value):
        if allow_none and (value is None or isinstance(value, str) and value.lower() == "none"):
            return None
        try:
            value = typeof(value)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Could not convert to {typeof.__name__}") from err
        if isinstance(value, str):
            value = value.lower()

        if value in accepted_values:
            # Convert value to python boolean if string matches
            value = {"true": True, "false": False}.get(value, value)
            return value
        raise ValueError(
            f'{value} is not one of {accepted_values}{" nor None" if allow_none else ""}'
        )

    return validate_choice


def _validate_positive_int(value):
    """Validate value is a natural number."""
    try:
        value = int(value)
    except ValueError as err:
        raise ValueError("Could not convert to int") from err
    if value > 0:
        return value
    raise ValueError("Only positive values are valid")


def _validate_float(value):
    """Validate value is a float."""
    try:
        value = float(value)
    except ValueError as err:
        raise ValueError("Could not convert to float") from err
    return value


def _validate_probability(value):
    """Validate a probability: a float between 0 and 1."""
    value = _validate_float(value)
    if (value < 0) or (value > 1):
        raise ValueError("Only values between 0 and 1 are valid.")
    return value


def _validate_boolean(value):
    """Validate value is a float."""
    if isinstance(value, str):
        value = value.lower()
    if value not in {True, False, "true", "false"}:
        raise ValueError("Only boolean values are valid.")
    return value is True or value == "true"


def _add_none_to_validator(base_validator):
    """Create a validator function that catches none and then calls base_fun."""
    # no blank lines allowed after function docstring by pydocstyle,
    # but black requires white line before function

    def validate_with_none(value):
        if value is None or isinstance(value, str) and value.lower() == "none":
            return None
        return base_validator(value)

    return validate_with_none


def make_iterable_validator(scalar_validator, length=None, allow_none=False, allow_auto=False):
    """Validate value is an iterable datatype."""
    # based on matplotlib's _listify_validator function

    def validate_iterable(value):
        if allow_none and (value is None or isinstance(value, str) and value.lower() == "none"):
            return None
        if isinstance(value, str):
            if allow_auto and value.lower() == "auto":
                return "auto"
            value = tuple(v.strip("([ ])") for v in value.split(",") if v.strip())
        if np.iterable(value) and not isinstance(value, set | frozenset):
            val = tuple(scalar_validator(v) for v in value)
            if length is not None and len(val) != length:
                raise ValueError(f"Iterable must be of length: {length}")
            return val
        raise ValueError("Only ordered iterable values are valid")

    return validate_iterable


defaultParams = {  # pylint: disable=invalid-name
    "stats.ci_kind": ("hdi", _make_validate_choice({"eti", "hdi"})),
    "stats.ci_prob": (0.94, _validate_probability),
    "plots.show_plot": (True, _validate_boolean),
}


class RcParams(MutableMapping):
    """Class to contain PreliZ default parameters.

    It is implemented as a dict with validation when setting items.
    """

    validate = {key: validate_fun for key, (_, validate_fun) in defaultParams.items()}

    # validate values on the way in
    def __init__(self, *args, **kwargs):
        self._underlying_storage = {}
        super().__init__()
        self.update(*args, **kwargs)

    def __setitem__(self, key, val):
        """Add validation to __setitem__ function."""
        try:
            try:
                cval = self.validate[key](val)
            except ValueError as verr:
                raise ValueError(f"Key {key}: {str(verr)}") from verr
            self._underlying_storage[key] = cval
        except KeyError as err:
            raise KeyError(
                f"{key} is not a valid rc parameter "
                f"(see rcParams.keys() for a list of valid parameters)"
            ) from err

    def __getitem__(self, key):
        """Use underlying dict's getitem method."""
        return self._underlying_storage[key]

    def __delitem__(self, key):
        """Raise TypeError if someone ever tries to delete a key from RcParams."""
        raise TypeError("RcParams keys cannot be deleted")

    def clear(self):
        """Raise TypeError if someone ever tries to delete all keys from RcParams."""
        raise TypeError("RcParams keys cannot be deleted")

    def pop(self, key, default=None):
        """Raise TypeError if someone ever tries to delete a key from RcParams."""
        raise TypeError(
            "RcParams keys cannot be deleted. Use .get(key) of RcParams[key] to check values"
        )

    def popitem(self):
        """Raise TypeError if someone ever tries to delete a key from RcParams."""
        raise TypeError(
            "RcParams keys cannot be deleted. Use .get(key) of RcParams[key] to check values"
        )

    def setdefault(self, key, default=None):
        """Raise error when using setdefault, defaults are handled on initialization."""
        raise TypeError(
            "Defaults in RcParams are handled on object initialization during library"
            "import. Use PreliZrc file instead."
            ""
        )

    def __repr__(self):
        """Customize repr of RcParams objects."""
        class_name = self.__class__.__name__
        indent = len(class_name) + 1
        repr_split = pprint.pformat(
            self._underlying_storage,
            indent=1,
            width=80 - indent,
        ).split("\n")
        repr_indented = ("\n" + " " * indent).join(repr_split)
        return f"{class_name}({repr_indented})"

    def __str__(self):
        """Customize str/print of RcParams objects."""
        return "\n".join(
            map(
                "{0[0]:<22}: {0[1]}".format,  # pylint: disable=consider-using-f-string
                sorted(self._underlying_storage.items()),
            )
        )

    def __iter__(self):
        """Yield sorted list of keys."""
        yield from sorted(self._underlying_storage.keys())

    def __len__(self):
        """Use underlying dict's len method."""
        return len(self._underlying_storage)

    def find_all(self, pattern):
        """
        Find keys that match a regex pattern.

        Return the subset of this RcParams dictionary whose keys match,
        using :func:`re.search`, the given ``pattern``.

        Notes
        -----
            Changes to the returned dictionary are *not* propagated to
            the parent RcParams dictionary.
        """
        pattern_re = re.compile(pattern)
        return RcParams((key, value) for key, value in self.items() if pattern_re.search(key))

    def copy(self):
        """Get a copy of the RcParams object."""
        return dict(self._underlying_storage)


def get_preliz_rcfile():
    """Get PreliZrc file.

    The file location is determined in the following order:

    - ``$PWD/PreliZrc``
    - ``$ARVIZ_DATA/prelizrc``
    - On Linux,
        - ``$XDG_CONFIG_HOME/preliz/prelizrc`` (if ``$XDG_CONFIG_HOME``
          is defined)
        - or ``$HOME/.config/preliz/prelizrc`` (if ``$XDG_CONFIG_HOME``
          is not defined)
    - On other platforms,
        - ``$HOME/.preliz/prelizrc`` if ``$HOME`` is defined

    Otherwise, the default defined in ``rcparams.py`` file will be used.
    """
    # no blank lines allowed after function docstring by pydocstyle,
    # but black requires white line before function

    def gen_candidates():
        yield os.path.join(os.getcwd(), "prelizrc")
        preliz_data_dir = os.environ.get("ARVIZ_DATA")
        if preliz_data_dir:
            yield os.path.join(preliz_data_dir, "prelizrc")
        xdg_base = os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))
        if sys.platform.startswith(("linux", "freebsd")):
            configdir = str(Path(xdg_base, "preliz"))
        else:
            configdir = str(Path.home() / ".preliz")
        yield os.path.join(configdir, "prelizrc")

    for fname in gen_candidates():
        if os.path.exists(fname) and not os.path.isdir(fname):
            return fname

    return None


def read_rcfile(fname):
    """Return :class:`preliz.RcParams` from the contents of the given file.

    Unlike `rc_params_from_file`, the configuration class only contains the
    parameters specified in the file (i.e. default values are not filled in).
    """
    _error_details_fmt = 'line #%d\n\t"%s"\n\tin file "%s"'

    config = RcParams()
    with open(fname, encoding="utf8") as rcfile:
        try:
            for line_no, line in enumerate(rcfile, 1):
                strippedline = line.split("#", 1)[0].strip()
                if not strippedline:
                    continue
                tup = strippedline.split(":", 1)
                if len(tup) != 2:
                    error_details = _error_details_fmt % (line_no, line, fname)
                    warnings.warn("Illegal %s", error_details)
                    continue
                key, val = tup
                key = key.strip()
                val = val.strip()
                if key in config:
                    warnings.warn("Duplicate key in file %r line #%d.", fname, line_no)
                try:
                    config[key] = val
                except ValueError as verr:
                    error_details = _error_details_fmt % (line_no, line, fname)
                    raise ValueError(f"Bad val {val} on {error_details}\n\t{str(verr)}") from verr

        except UnicodeDecodeError:
            warnings.warn(
                "Cannot decode configuration file %s with encoding "
                "%s, check LANG and LC_* variables.",
                fname,
                locale.getpreferredencoding(do_setlocale=False) or "utf-8 (default)",
            )
            raise

        return config


def rc_params(ignore_files=False):
    """Read and validate PreliZrc file."""
    fname = None if ignore_files else get_preliz_rcfile()
    defaults = RcParams([(key, default) for key, (default, _) in defaultParams.items()])
    if fname is not None:
        file_defaults = read_rcfile(fname)
        defaults.update(file_defaults)
    return defaults


rcParams = rc_params()  # pylint: disable=invalid-name


class rc_context:  # pylint: disable=invalid-name
    """
    Return a context manager for managing rc settings.

    Parameters
    ----------
    rc : dict, optional
        Mapping containing the rcParams to modify temporally.
    fname : str, optional
        Filename of the file containing the rcParams to use inside the rc_context.

    Examples
    --------
    This allows one to do::

        with pz.rc_context({"plots.show_plot": False}):
            pz.maxent(pz.Normal())

    The 'rc' dictionary takes precedence over the settings loaded from
    'fname'.
    """

    def __init__(self, rc=None, fname=None):
        self._orig = rcParams.copy()
        if fname:
            file_rcparams = read_rcfile(fname)
            rcParams.update(file_rcparams)
        if rc:
            rcParams.update(rc)

    def __enter__(self):
        """Define enter method of context manager."""
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Define exit method of context manager."""
        rcParams.update(self._orig)
