"""
PreliZ.

Tools to help you pick a prior
"""
import logging

from matplotlib import rcParams

from .maxent import maxent
from .ppa import ppa
from .roulette import roulette
from .quartile import quartile
from .mle import mle
from .distributions import *

__all__ = ["maxent", "mle", "ppa", "roulette", "quartile"]

__version__ = "0.0.2"

_log = logging.getLogger("preliz")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)

# Allow legend outside plot in maxent to be included when saving a figure
# We may want to make this more explicit by having preliz.rcParams
rcParams["savefig.bbox"] = "tight"
