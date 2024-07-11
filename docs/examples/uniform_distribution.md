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
# Uniform Distribution

The Uniform distribution is a continuous probability distribution bounded between two real numbers, $lower$ and $upper$, representing the lower and upper bounds, respectively.

The probability density of the Uniform distribution is constant between $lower$ and $upper$ and zero elsewhere. Despite its simplicity, it is rare to observe a real-world distribution where all outcomes are equally likely.

The Uniform distribution is the maximum entropy probability distribution for a random variable under no constraint other than that it is contained in the interval $[lower,upper]$. It's often employed in random sampling, for generating random numbers, and as a non-informative (flat) prior in Bayesian statistics when there is no prior knowledge about the parameter other than its range.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Uniform Distribution PDF
---

import matplotlib.pyplot as plt
import arviz as az
from preliz import Uniform
az.style.use('arviz-doc')
ls = [1, -2]
us = [6, 2]
for l, u in zip(ls, us):
    ax = Uniform(l, u).plot_pdf()
ax.set_ylim(0, 0.3)

```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Uniform Distribution CDF
---

for l, u in zip(ls, us):
    Uniform(l, u).plot_cdf()
```

## Key properties and parameters:

```{eval-rst}
========  =====================================
Support   :math:`x \in [lower, upper]`
Mean      :math:`\dfrac{lower + upper}{2}`
Variance  :math:`\dfrac{(upper - lower)^2}{12}`
========  =====================================
```

**Probability Density Function (PDF):**

$$
f(x \mid lower, upper) =
    \begin{cases}
        \dfrac{1}{upper - lower} & \text{for } x \in [lower, upper] \\
        0 & \text{otherwise}
    \end{cases}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x \mid lower, upper) =
    \begin{cases}
        0 & \text{for } x < lower \\
        \dfrac{x - lower}{upper - lower} & \text{for } x \in [lower, upper] \\
        1 & \text{for } x > upper
    \end{cases}
$$

```{seealso}
:class: seealso

**Common Alternatives:**

- [Beta Distribution](beta_distribution.md) - The Uniform distribution is a special case of the Beta distribution with $\alpha = \beta = 1$.

**Related Distributions:**

- [Triangular Distribution](triangular_distribution.md) - It is defined between a range of values, similar to the Uniform distribution, but with a peak at one of the points.
```

# References

- Wikipedia - [Uniform distribution](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)



