"""
PreliZ.

Tools to help you pick a prior
"""
import logging

from .constraints import constraints
from .ppa import ppa
from .roulette import roulette

__all__ = ["roulette", "constraints", "ppa"]

__version__ = "0.0.1.dev0"

_log = logging.getLogger("preliz")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
