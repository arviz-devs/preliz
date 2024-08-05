"""
PreliZ.

Tools to help you pick a prior
"""
import logging
from os import path as os_path

from matplotlib import rcParams
from matplotlib import style

from .distributions import *
from .predictive import *
from .unidimensional import *


__all__ = ["maxent", "mle", "ppa", "roulette", "quartile"]

__version__ = "0.8.1"

_log = logging.getLogger("preliz")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

# Allow legend outside plot in maxent to be included when saving a figure
# We may want to make this more explicit by having preliz.rcParams
rcParams["savefig.bbox"] = "tight"


# add PreliZ's styles to matplotlib's styles
_preliz_style_path = os_path.join(os_path.dirname(__file__), "styles")
style.core.USER_LIBRARY_PATHS.append(_preliz_style_path)
style.core.reload_library()
