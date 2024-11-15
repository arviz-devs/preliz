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
# Discrete Uniform Distribution

<audio controls> <source src="../../_static/discreteuniform.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Discrete Uniform distribution is a probability distribution where each integer value between `lower` and `upper` (inclusive) has the same probability. This distribution is characterized by two parameters: `lower` and `upper`, defining the range of integers.

A simple example of the Discrete Uniform distribution is rolling a fair six-sided die, where each face has an equal probability of 1/6.

## Probability Mass Function (PMF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Discrete Uniform Distribution PMF
---

from preliz import DiscreteUniform, style
style.use('preliz-doc')
ls = [1, -2]
us = [6, 2]
for l, u in zip(ls, us):
    DiscreteUniform(l, u).plot_pdf()
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Discrete Uniform Distribution CDF
---

for l, u in zip(ls, us):
    DiscreteUniform(l, u).plot_cdf()
```
## Key properties and parameters:

```{eval-rst}
========  ============================================================
Support   :math:`x \in \{ \text{lower}, \text{lower} + 1, \ldots, \text{upper} \}`
Mean      :math:`\dfrac{\text{lower} + \text{upper}}{2}`
Variance  :math:`\dfrac{(\text{upper} - \text{lower} + 1)^2 - 1}{12}`
========  ============================================================
```

**Probability Mass Function (PMF):**

$$
f(x \mid lower, upper) = \frac{1}{upper-lower+1}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x \mid lower, upper) = = \frac{x - lower + 1}{upper - lower + 1}
$$

```{seealso}
:class: seealso

** Common Alternatives:**

- [Categorical](categotical.md) - The Discrete Uniform distribution is a special case of the Categorical distribution  where all elements of $p$ are equal.

**Related Distributions:**

- [Uniform](uniform.md) - The continuous version of the Discrete Uniform distribution.
```

## References

- [Wikipedia - Discrete Uniform Distribution](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)

