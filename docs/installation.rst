Installation
============

PreliZ is tested on Python 3.11+. And depends on ArviZ, matplotlib, NumPy, and SciPy. See `pyproject.toml <https://github.com/arviz-devs/preliz/blob/main/pyproject.toml>`_ for version information.


For the latest release, you can install PreliZ either using pip or conda-forge:

Using pip
---------

.. code-block:: bash

  pip install preliz

To make use of the interactive features, you can install the optional dependencies:

.. tabs::

  .. tab:: JupyterLab
    .. code-block:: bash

      pip install "preliz[full,lab]"

  .. tab:: Jupyter Notebook
    .. code-block:: bash

      pip install "preliz[full,notebook]"

Using conda-forge
-----------------

.. code-block:: bash

    conda install -c conda-forge preliz


Development version
-------------------

The latest development version can be installed from the main branch using pip:

.. code-block:: bash

  pip install git+https://github.com/arviz-devs/preliz.git
