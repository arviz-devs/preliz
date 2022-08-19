"""
PreliZ.

Tools to help you define priors
"""
import logging

from matplotlib import rcParams

from .maxent import maxent
from .ppa import ppa
from .roulette import roulette
from .quartile import quartile
from .distributions import *

__all__ = ["maxent", "ppa", "roulette", "quartile"]

__version__ = "0.0.1.dev0"

_log = logging.getLogger("preliz")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        _log.addHandler(logging.StreamHandler())

# Allow legend outside plot in maxent to be included when saving a figure
# We may want to make this more explicit by having preliz.rcParams
rcParams["savefig.bbox"] = "tight"


# clean namespace
del logging, rcParams
