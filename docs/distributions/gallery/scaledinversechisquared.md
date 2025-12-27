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
# Scaled Inverse Chi-Squared Distribution

<audio controls> <source src="../../_static/scaledinversechisquared.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Non-Negative](../../gallery_tags.rst#non-negative)

The Scaled Inverse chi-squared (Scale-Inv-$\chi^2$) is a continuous probability distribution characterized by two parameters, $\nu$, usually called degrees of freedom, and $\tau^2$, a scale parameter.

The Scaled Inverse chi-squared distribution is mainly used in Bayesian inference as a conjugate prior for variance parameters in normal models, particularly in Bayesian regression, hierarchical models, 
and time-series analysis.

## Key properties and parameters

```{eval-rst}
========  ==============================================================
Support   :math:`x \in [0, \infty)`
Mean      :math:`\nu \tau^2 / (\nu - 2)` for :math:`\nu > 2`, else :math:`\infty`
Variance  :math:`\frac{2 \nu^2 \tau^4}{(\nu - 2)^2 (\nu - 4)}`
          for :math:`\nu > 4`, else :math:`\infty`
========  ==============================================================
```

**Parameters:**

- $\nu$ : (float) Degrees of freedom, $\nu > 0$.
- $\tau^2$ : (float) Scale parameter, $\tau^2 > 0$.

### Probability Density Function (PDF)

$$
f(x \mid \nu, \tau^2)
=
\frac{(\tau^2 \nu / 2)^{\nu/2}}{\Gamma(\nu/2)}
\
\frac{\exp\left[-\dfrac{\nu \tau^2}{2x}\right]}{x^{1+\nu/2}}
$$

where $\Gamma(\nu)$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Scaled Inverse Chi-Squared Distribution PDF
---

import numpy as np
from preliz import ScaledInverseChiSquared, style
style.use('preliz-doc')

nus = [1, 5, 10]
tau2s = [1, 1, 1]

for nu, tau2 in zip(nus,tau2s):
    ax = ScaledInverseChiSquared(nu, tau2).plot_pdf(support=(np.finfo(float).eps, 5))
    ax.set_ylim(0, 1)
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid \nu, \tau^2) = \frac{\Gamma\left(\dfrac{\nu}{2}, \dfrac{\tau^2\nu}{2x}\right)}{\Gamma\left(\dfrac{\nu}{2}\right)}
$$

where $\Gamma(\nu, x)$ is the [incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Scaled Inverse Chi-Squared Distribution CDF
---

for nu, tau2 in zip(nus,tau2s):
    ax = ScaledInverseChiSquared(nu, tau2).plot_cdf(support=(np.finfo(float).eps, 5))
    ax.set_ylim(0, 1)
```

```{seealso}
:class: seealso

**Related Distributions:**

- [Chi-Squared](chisquared.md) - The Inverse Chi-Squared distribution can be thought of as the reciprocal of a scaled Chi-Squared random variable.
- [Gamma](gamma.md) - the Inverse Chi-Squared distribution can be seen as the reciprocal of a Gamma-distributed variable with specific parameters.
```

## References

- [Wikipedia - Scaled Inverse Chi-squared distribution](https://en.wikipedia.org/wiki/Scaled_inverse_chi-squared_distribution)

