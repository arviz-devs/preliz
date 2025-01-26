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
# Von Mises Distribution

<audio controls> <source src="../../_static/vonmises.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Von Mises distribution is a continuous probability distribution on the unit circle. It is characterized by two parameters: $\mu$ and $\kappa$, which are the mean direction and concentration parameter, respectively.

The Von Mises distribution is the circular analogue of the normal distribution, and it is used to model circular data, such as wind directions, compass bearings, or angles. 

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in (-\pi, \pi)`
Mean      :math:`\mu`
Variance  :math:`1 - I_1(\kappa) / I_0(\kappa)`
========  ==========================================
```

**Parameters:**

- $\mu$ : (float) Mean direction, $-\pi \leq \mu \leq \pi$.
- $\kappa$ : (float) Concentration parameter, $\kappa \geq 0$.

### Probability Density Function (PDF)

$$
f(x|\mu, \kappa) = \frac{e^{\kappa \cos(x - \mu)}}{2\pi I_0(\kappa)}
$$

where $I_0(\kappa)$ is the [modified Bessel function of the first kind](https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1).

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Von Mises Distribution PDF
---
import numpy as np

from preliz import style, VonMises
style.use('preliz-doc')
mus = [0., 0., 0.,  -2.5]
kappas = [.01, 0.5, 4., 2.]
for mu, kappa in zip(mus, kappas):
    VonMises(mu, kappa).plot_pdf(support=(-np.pi,np.pi))
```

### Cumulative Distribution Function (CDF)

The Von Mises distribution does not have an analytical expression for the CDF. However, it can be evaluated numerically by integrating the PDF in the interval $(-\pi, x)$:

$$
F(x|\mu, \kappa) = \frac{1}{2\pi I_0(\kappa)} \int_{-\pi}^{x} e^{\kappa \cos(t - \mu)} dt
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Von Mises Distribution CDF
---

for mu, kappa in zip(mus, kappas):
    VonMises(mu, kappa).plot_cdf(support=(-np.pi,np.pi))
```

```{seealso}
:class: seealso

**Related Distributions:**

- [Normal Distribution](normal.md) - When $\kappa \to \infty$, the Von Mises distribution approximates the normal distribution.
- [Uniform Distribution](uniform.md) - When $\kappa = 0$, the Von Mises distribution converges to the uniform distribution in the interval $(-\pi, \pi)$.
```

## References

- [Wikipedia - Von Mises distribution](https://en.wikipedia.org/wiki/Von_Mises_distribution)