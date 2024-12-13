"""
PreliZ.

Tools to help you pick a prior
"""
from os import path as os_path

from matplotlib import rcParams
from matplotlib import style

from .distributions import *
from .predictive import *
from .ppls import *
from .unidimensional import *
from .multidimensional import *


__version__ = "0.13.0"


# Allow legend outside plot in maxent to be included when saving a figure
# We may want to make this more explicit by having preliz.rcParams
rcParams["savefig.bbox"] = "tight"


# add PreliZ's styles to matplotlib's styles
_preliz_style_path = os_path.join(os_path.dirname(__file__), "styles")
style.core.USER_LIBRARY_PATHS.append(_preliz_style_path)
style.core.reload_library()

# clean namespace
del os_path, rcParams, _preliz_style_path
