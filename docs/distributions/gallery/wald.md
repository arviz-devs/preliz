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
# Wald Distribution

<audio controls> <source src="../../_static/wald.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Wald distribution, also known as the Inverse Gaussian distribution, is a continuous probability distribution characterized by its positive support and skewed shape. It is defined by two parameters: the mean ($\mu$) and the scale ($\lambda$). It is also used in survival analysis to model the time to an event.

The "Inverse" in the name can be misleading. The Wald distribution describes the time a particle subject to Brownian motion will drift to a certain point. Meanwhile, the Gaussian distribution gives the position of the motion at a fixed time. Only in that sense, the Wald is the "inverse" of the Gaussian.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in (0, \infty)`
Mean      :math:`\mu`
Variance  :math:`\mu^3 / \lambda`
========  ==========================================
```

**Parameters:**

- $\mu$ : Mean of the distribution.
- $\lambda$ : Scale parameter.

**Alternative parametrizations:**

Wald distribution has 3 alternative parametrizations. In terms of $\mu$ and $\lambda$, $\mu$ and $\phi$, and $\lambda$ and $\phi$.

The link between the 3 alternatives is given by:

$$
\phi = \frac{\lambda}{\mu} \\
$$

### Probability Density Function (PDF)

$$
f(x \mid \mu, \lambda) = \left(\frac{\lambda}{2\pi}\right)^{1/2} x^{-3/2} \exp\left\{-\frac{\lambda}{2x}\left(\frac{x-\mu}{\mu}\right)^2    \right\}
$$

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$ and $\lambda$
:sync: mu-lambda
```{jupyter-execute}
:hide-code:

from preliz import Wald, style
style.use('preliz-doc')
mus = [1., 1., 1., 4.]
lams = [1., 0.2, 3., 1.]
for mu, lam in zip(mus, lams):
    Wald(mu, lam).plot_pdf(support=(0, 4))
```
:::::

:::::{tab-item} Parameters $\mu$ and $\phi$
:sync: mu-phi

```{jupyter-execute}
:hide-code:

phis = [1., 0.2, 3., 0.25]
for mu, phi in zip(mus, phis):
    Wald(mu, phi=phi).plot_pdf(support=(0, 4))
```
:::::

:::::{tab-item} Parameters $\lambda$ and $\phi$
:sync: lambda-phi

```{jupyter-execute}
:hide-code:

for lam, phi in zip(lams, phis):
    Wald(lam=lam, phi=phi).plot_pdf(support=(0, 4))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x \mid \mu, \lambda) = \Phi\left(\sqrt{\frac{\lambda}{x}}\left(\frac{x}{\mu} - 1\right)\right)
$$

where $\Phi$ is the standard normal cumulative distribution function.

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$ and $\lambda$
:sync: mu-lambda
```{jupyter-execute}
:hide-code:

for mu, lam in zip(mus, lams):
    Wald(mu, lam).plot_cdf(support=(0, 4))
```
:::::

:::::{tab-item} Parameters $\mu$ and $\phi$
:sync: mu-phi

```{jupyter-execute}
:hide-code:

for mu, phi in zip(mus, phis):
    Wald(mu, phi=phi).plot_cdf(support=(0, 4))
```
:::::

:::::{tab-item} Parameters $\lambda$ and $\phi$
:sync: lambda-phi

```{jupyter-execute}
:hide-code:

for lam, phi in zip(lams, phis):
    Wald(lam=lam, phi=phi).plot_cdf(support=(0, 4))
```
:::::
::::::

```{seealso}
:class: seealso

**Related Distributions:**

- [Inverse Gamma Distribution](inversegamma.md) - When $\mu \to \infty$ (zero drift velocity in Brownian motion), the Wald distribution converges to the inverse gamma distribution.
```

## References

- Wikipedia - [Wald Distribution](https://en.wikipedia.org/wiki/Inverse_Gaussian_distribution)