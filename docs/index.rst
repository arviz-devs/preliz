A tool-box for prior elicitation 
===================================================

|PyPI version|
|Tests|
|Coverage|
|Black|

.. |PyPI version| image:: https://badge.fury.io/py/preliz.svg
    :target: https://badge.fury.io/py/preliz
  
.. |Tests| image:: https://github.com/arviz-devs/preliz/actions/workflows/test.yml/badge.svg
    :target: https://github.com/arviz-devs/preliz

.. |Coverage| image:: https://codecov.io/gh/arviz-devs/preliz/branch/main/graph/badge.svg?token=SLJIK2O4C5 
    :target: https://codecov.io/gh/arviz-devs/preliz

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/ambv/black


Overview
========

Prior elicitation refers to the process of transforming the knowledge of a particular domain into well-defined probability distributions.
Specifying useful priors is a central aspect of Bayesian statistics. PreliZ is a Python package aimed at helping practitioners choose prior
distributions by offering a set of tools for the various facets of prior elicitation. It covers a range of methods, from unidimensional prior
elicitation on the parameter space to predictive elicitation on the observed space. The goal is to be compatible with probabilistic programming
languages (PPL) in the Python ecosystem like PyMC and PyStan, while remaining agnostic of any specific PPL.


Dependencies
============
PreliZ is tested on Python 3.8+.

Installation
============

Two dependency bundles are available for the latest release:

* ``full``: includes dependencies for all features (interactive and non-interactive)
* ``non-interactive``: includes dependencies for non-interactive features only.

You can install them with:

.. tabs::

  .. tab:: Full
    .. code-block:: bash

      pip install "preliz[full]"

  .. tab:: Non-interactive
    .. code-block:: bash

      pip install "preliz[non-interactive]"

The latest development version can be installed from the main branch using pip:

.. code-block:: bash

  pip install git+https://github.com/arviz-devs/preliz.git

Contributing
============
We welcome contributions from interested individuals or groups! For information about contributing to PreliZ check out our instructions, policies, and guidelines `here <https://github.com/arviz-devs/preliz/blob/main/CONTRIBUTING.md>`_.

Contributors
============
See the `GitHub contributor page <https://github.com/arviz-devs/preliz/graphs/contributors>`_.

Contents
========

.. toctree::
   :maxdepth: 2

   examples/param_space_1d_examples
   examples/observed_space_examples

   api_reference

Indices
=======

* :ref:`genindex`
* :ref:`modindex`
