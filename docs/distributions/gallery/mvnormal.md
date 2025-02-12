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
# Multivariate Normal Distribution

<audio controls> <source src="../../_static/multivariatenormal.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Multivariate](../../gallery_tags.rst#multivariate), [Continuous](../../gallery_tags.rst#continuous), [Symmetric](../../gallery_tags.rst#symmetric), [Unbounded](../../gallery_tags.rst#unbounded)

The multivariate normal distribution, also known as the multivariate Gaussian distribution, is a generalization of the univariate normal distribution to multiple dimensions. A random vector is said to follow a $k$-dimensional multivariate normal distribution if every linear combination of its $k$ components follows a univariate normal distribution.

The multivariate normal distribution is often used to describe the joint distribution of a set of correlated random variables. 

In Bayesian modeling, a Gaussian process is a generalization of the multivariate normal distribution to infinite dimensions, and it is used as a prior distribution over functions.


## Key properties and parameters

```{eval-rst}
========  ==========================
Support   :math:`x \in \mathbb{R}^k`
Mean      :math:`\mu`
Variance  :math:`T^{-1}`
========  ==========================
```

**Parameters:**

- $\mu$ : (array of floats) Mean vector of length $k$.
- $\Sigma$ : (array of floats) Covariance matrix of shape $k \times k$.
- $T$ : (array of floats) Precision matrix, the inverse of the covariance matrix.

**Alternative parameterization:**

The MvNormal has 2 alternative parameterizations. In terms of the mean and the covariance matrix, or in terms of the mean and the precision matrix.

The link between the 2 alternatives is given by:

$$
T = \Sigma^{-1}
$$

### Probability Density Function (PDF)

$$
f(x \mid \mu, T) = \frac{|T|^{1/2}}{(2\pi)^{k/2}} \exp\left\{ -\frac{1}{2} (x-\mu)^{\prime} T (x-\mu) \right\}
$$

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$ and $\Sigma$
:sync: mu-sigma
```{jupyter-execute}
:hide-code:

import matplotlib.pyplot as plt
import numpy as np
from preliz import MvNormal

_, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
mus = [[0., 0], [3, -2], [0., 0], [0., 0]]
sigmas = [np.eye(2), np.eye(2), np.array([[2, 2], [2, 4]]), np.array([[2, -2], [-2, 4]])]
for mu, sigma, ax in zip(mus, sigmas, axes.ravel()):
    MvNormal(mu, sigma).plot_pdf(marginals=False, ax=ax)
```
:::::

:::::{tab-item} Parameters $\mu$ and $T$
:sync: mu-t
```{jupyter-execute}
:hide-code:

_, axes = plt.subplots(2, 2, figsize=(9, 9), sharex=True, sharey=True)
mus = [[0., 0], [3, -2], [0., 0], [0., 0]]
Ts = [np.linalg.inv(sigma) for sigma in sigmas]
for mu, T, ax in zip(mus, Ts, axes.ravel()):
    MvNormal(mu, tau=T).plot_pdf(marginals=False, ax=ax)
```
:::::
::::::

### Cumulative Distribution Function (CDF)

The multivariate normal joint CDF does not have a closed-form analytic solution in the general case, due to the complexity of integrating its density over $\mathbb{R}^k$. However, each marginal distribution is a univariate normal distribution with a well-defined CDF that can be computed directly.

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$ and $\Sigma$
:sync: mu-sigma
```{jupyter-execute}
:hide-code:

for mu, sigma in zip(mus, sigmas):
    MvNormal(mu, sigma).plot_cdf()
```
:::::

:::::{tab-item} Parameters $\mu$ and $T$
:sync: mu-t
```{jupyter-execute}
:hide-code:

for mu, T in zip(mus, Ts):
    MvNormal(mu, tau=T).plot_cdf()
```
:::::
::::::

```{seealso}
:class: seealso

**Related Distributions:**

- [Normal Distribution](normal.md) - The univariate normal distribution is a special case of the multivariate normal distribution with $k=1$.
```

## References

- [Wikipedia - Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
