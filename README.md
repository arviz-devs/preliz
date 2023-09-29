<img src="https://raw.githubusercontent.com/arviz-devs/preliz/main/docs/logos/PreliZ.png#gh-light-mode-only" width=200></img>
<img src="https://raw.githubusercontent.com/arviz-devs/preliz/main/docs/logos/PreliZ_white.png#gh-dark-mode-only" width=200></img>

A tool-box for prior elicitation.

[![PyPi version](https://badge.fury.io/py/preliz.svg)](https://badge.fury.io/py/preliz)
[![Build Status](https://github.com/arviz-devs/preliz/actions/workflows/test.yml/badge.svg)](https://github.com/arviz-devs/preliz/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/arviz-devs/preliz/branch/master/graph/badge.svg?token=SLJIK2O4C5 )](https://codecov.io/gh/arviz-devs/preliz/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05499/status.svg)](https://doi.org/10.21105/joss.05499)

## Overview

Prior elicitation refers to the process of transforming the knowledge of a particular domain into well-defined probability distributions. Specifying useful priors is a central aspect of Bayesian statistics. PreliZ is a Python package aimed at helping practitioners choose prior distributions by offering a set of tools for the various facets of prior elicitation. It covers a range of methods, from unidimensional prior elicitation on the parameter space to predictive elicitation on the observed space. The goal is to be compatible with probabilistic programming languages (PPL) in the Python ecosystem like PyMC and PyStan, while remaining agnostic of any specific PPL.

###  The Zen of PreliZ
* Being open source, community-driven, diverse and inclusive.
* Avoid fully-automated solutions, keep the human in the loop.
* Separate tasks between humans and computers, so users can retain control of important decisions while numerically demanding, error-prone or tedious tasks are automatized.
* Prevent users to become overconfident in their own opinions.
* Easily integrate with other tools.
* Allow predictive elicitation.
* Having a simple and intuitive interface suitable for non-specialists in order to minimize cognitive biases and heuristics.
* Switching between different types of visualization such as kernel density estimates plots, quantile dotplots, histograms, etc. 
* Being agnostic of the underlying probabilistic programming language.
* Being modular.


## Documentation

The PreliZ documentation can be found in the [official docs](https://preliz.readthedocs.io/en/latest/).

## Installation

### Last release
PreliZ is available for installation from [PyPI](https://pypi.org/project/preliz/).
The latest version (base set of dependencies) can be installed using pip:

```
pip install preliz
```
To make use of the interactive features, you can install the optional dependencies:

* For JupyterLab:

```
pip install "preliz[full,lab]"
```

* For Jupyter Notebook:

```
pip install "preliz[full,notebook]"
```

### Development
The latest development version can be installed from the main branch using pip:

```
pip install git+git://github.com/arviz-devs/preliz.git
```

## Citation
If you find PreliZ useful in your work, we kindly request that you cite the following paper:

```
@article{Icazatti_2023,
author = {Icazatti, Alejandro and Abril-Pla, Oriol and Klami, Arto and Martin, Osvaldo A},
doi = {10.21105/joss.05499},
journal = {Journal of Open Source Software},
month = sep,
number = {89},
pages = {5499},
title = {{PreliZ: A tool-box for prior elicitation}},
url = {https://joss.theoj.org/papers/10.21105/joss.05499},
volume = {8},
year = {2023}
}
```

## Contributions
PreliZ is a community project and welcomes contributions.
Additional information can be found in the [Contributing Readme](https://github.com/arviz-devs/preliz/blob/main/CONTRIBUTING.md)


## Code of Conduct
PreliZ wishes to maintain a positive community. Additional details
can be found in the [Code of Conduct](https://github.com/arviz-devs/preliz/blob/main/CODE_OF_CONDUCT.md)

## Donations
PreliZ, as other ArviZ-devs projects, is a non-profit project under the NumFOCUS umbrella. If you want to support PreliZ financially, you can donate [here](https://numfocus.org/donate-to-arviz).

## Sponsors
[![NumFOCUS](https://www.numfocus.org/wp-content/uploads/2017/07/NumFocus_LRG.png)](https://numfocus.org)
