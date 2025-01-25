---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Predictive Explorer

This function will automatically generate textboxes for a function, which makes it easier to explore how the prior
predictive distribution changes when we change the priors. This function supports **PreliZ**, **PyMC** and **Bambi**
Models.

```{jupyter-execute}

import preliz as pz
import numpy as np
import pandas as pd
```

```{jupyter-execute}

pz.style.library["preliz-doc"]["figure.dpi"] = 100
pz.style.library["preliz-doc"]["figure.figsize"] = (10, 4)
pz.style.use("preliz-doc")
```

::::::{tab-set}
:class: full-width

:::::{tab-item} PreliZ
:sync: preliz

Suppose you already have a model, but you are unsure about the implications of its parameters. You can write it using
PreliZ distributions and put it inside a function with the arguments being the parameters you want to explore.

```{jupyter-execute}

x = np.linspace(0, 1, 100)
def a_preliz_model(a_mu, a_sigma, c_sigma=1):
    a = pz.Normal(a_mu, a_sigma).rvs()
    c = pz.Gamma(mu=2, sigma=c_sigma).rvs()
    b = pz.Normal(np.exp(a)*x, c).rvs()
    return b
```

By calling `predictive_explorer` you will get textboxes with some default initial values and for you to explore.

```{jupyter-execute}

pz.predictive_explorer(a_preliz_model)
```

After the parameter name, you will see a tuple indicating the valid range of the parameter. These values are inferred automatically by `predictive_explorer`.

Currently, we use a very simple heuristic to find the range, so take the suggestion with a pinch of salt
`predictive_explorer` supports three types of plots: empirical cumulative distribution functions (CDFs), kernel density estimations (KDEs), and histograms. Additionally, you can also add custom Matplotlib code using the `plot_func` parameter.

```{jupyter-execute}

# Custom function to plot a histogram
def custom_hist(predictions, ax):
    ax.hist(predictions.flatten(), bins='auto', alpha=0.7)
```

```{jupyter-execute}

# plot_func set to custom_hist for preliz model
pz.predictive_explorer(a_preliz_model, samples=10, plot_func=custom_hist)
```
:::::

:::::{tab-item} PyMC
:sync: pymc

PyMC is an optional dependency of PreliZ, you only need it if you want to use the `predictive_explorer` function with PyMC models.

To install *PyMC*, you can run the following command:

```bash

conda install -c conda-forge pymc
```

```{jupyter-execute}

# Add pymc
import pymc as pm
```

You can write the model using *PyMC* distributions and place it inside the function with the arguments being the parameters you want to explore.

```{jupyter-execute}

x = np.linspace(0, 1, 100)
def a_pymc_model(a_mu, a_sigma, c_sigma=1):
    with pm.Model() as model:
        a = pm.Normal("a", a_mu, a_sigma)
        c = pm.Gamma("c", mu=2, sigma=c_sigma)
        b = pm.Normal("b", np.exp(a) * x, c, observed=[0] * 100)
    return model
```

The `predictive_explorer` function auto-detects that the model contains PyMC distributions, alternatively you can specify that the function should use the PyMC engine by providing the parameter `engine=pymc`.

```{jupyter-execute}

pz.predictive_explorer(a_pymc_model)
```

After the parameter name, you will see a tuple indicating the valid range of the parameter. These values are inferred automatically by `predictive_explorer`.

Currently, we use a very simple heuristic to find the range, so take the suggestion with a pinch of salt 
`predictive_explorer` supports three types of plots: empirical cumulative distribution functions (CDFs), kernel density estimations (KDEs), and histograms. Additionally, you can also add custom Matplotlib code using the ``plot_func` parameter

```{jupyter-execute}

# Custom function to plot a histogram
def custom_hist(predictions, ax):
    ax.hist(predictions.flatten(), bins='auto', alpha=0.7)
```

```{jupyter-execute}

# plot_func set to custom_hist for pymc model
pz.predictive_explorer(a_pymc_model, samples=10, plot_func=custom_hist)
```
:::::

:::::{tab-item} Bambi
:sync: bambi

Bambi is an optional dependency of PreliZ, you only need it if you want to use the ``predictive_explorer`` function with Bambi models.

To install *Bambi*, you can run the following command:

```bash

conda install -c conda-forge bambi
```

```{jupyter-execute}

# Add bambi
import bambi as bmb
```

The `predictive_explorer` function allows you to write the model using *Bambi* distributions and inout it inside the function with the arguments being the parameters you want to explore.

```{jupyter-execute}

data = pd.DataFrame(
{
    "y": np.random.normal(size=100),
    "x": np.random.normal(size=100),
}
)
def a_bambi_model(a_mu, a_sigma):
    prior = {"Intercept": bmb.Prior("Normal", mu=a_mu, sigma=a_sigma)}
    a_model = bmb.Model("y ~ x", data, priors=prior)
    return a_model
```

The `predictive_explorer` function automatically detects if the model contains Bambi distributions. Alternatively, you can specify that the function should use the Bambi engine by providing the parameter `engine=bambi`.

```{jupyter-execute}

pz.predictive_explorer(a_bambi_model)
```

After the parameter name, you will see a tuple indicating the valid range of the parameter. These values are inferred automatically by `predictive_explorer`.

Currently, we use a very simple heuristic to find the range, so take the suggestion with a pinch of salt
`predictive_explorer` supports three types of plots: empirical cumulative distribution functions (CDFs), kernel density estimations (KDEs), and histograms. Additionally, you can also add custom Matplotlib code using the `plot_func` parameter.

```{jupyter-execute}

# Custom function to plot a histogram
def custom_hist(predictions, ax):
    ax.hist(predictions.flatten(), bins='auto', alpha=0.7)
```

```{jupyter-execute}

pz.predictive_explorer(a_bambi_model, samples=10, plot_func=custom_hist)
```
