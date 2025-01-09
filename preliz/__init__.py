"""
PreliZ.

Exploring and eliciting probability distributions
"""
from os import path as os_path

from matplotlib import rcParams as mpl_rcParams
from matplotlib import style

from .distributions import *
from .predictive import *
from .unidimensional import *
from .multidimensional import *
from .internal.rcparams import rc_context, rcParams


__version__ = "0.15.0"


# Allow legend outside plot in maxent to be included in the saved figure
mpl_rcParams["savefig.bbox"] = "tight"


# add PreliZ's styles to matplotlib's styles
_preliz_style_path = os_path.join(os_path.dirname(__file__), "styles")
style.core.USER_LIBRARY_PATHS.append(_preliz_style_path)
style.core.reload_library()

# clean namespace
del os_path, mpl_rcParams, _preliz_style_path
