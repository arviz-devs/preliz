"""
PreliZ.

Tools to help you pick a prior
"""
import logging

from matplotlib import rcParams

from .constraints import constraints
from .ppa import ppa
from .roulette import roulette
from .distributions import *

__all__ = ["constraints", "ppa", "roulette"]

__version__ = "0.0.1.dev0"

_log = logging.getLogger("preliz")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

# Allow legend outside plot in constraints to be included when saving a figure
# We may want to make this more explicit by having preliz.rcParams
rcParams["savefig.bbox"] = "tight"
