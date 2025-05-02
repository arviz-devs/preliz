"""
PreliZ.

Exploring and eliciting probability distributions
"""

from os import path as os_path

from matplotlib import rcParams as mpl_rcParams
from matplotlib import style

from preliz.distributions import *
from preliz.predictive import *
from preliz.unidimensional import *
from preliz.multidimensional import *
from preliz.internal.rcparams import rc_context, rcParams
from preliz.internal.citations import citations


__version__ = "0.17.0"


# Allow legend outside plot in maxent to be included in the saved figure
mpl_rcParams["savefig.bbox"] = "tight"


# add PreliZ's styles to matplotlib's styles
_preliz_style_path = os_path.join(os_path.dirname(__file__), "styles")
style.core.USER_LIBRARY_PATHS.append(_preliz_style_path)
style.core.reload_library()

# clean namespace
del os_path, mpl_rcParams, _preliz_style_path
