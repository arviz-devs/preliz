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
# Categorical Distribution

<audio controls> <source src="../../_static/categorical.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Categorical distribution is the most general discrete distribution and is parameterized by a vector $p$ where each element $p_i$ specifies the probabilities of each possible outcome.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Categorical Distribution PDF
---

from preliz import Categorical, style
style.use('preliz-doc')
ps = [[0.1, 0.6, 0.3], [0.3, 0.1, 0.1, 0.5]]
for p in ps:
    Categorical(p).plot_pdf()
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Categorical Distribution CDF
---

for p in ps:
    Categorical(p).plot_cdf()
```

## Key properties and parameters:

```{eval-rst}
========  ===================================
Support   :math:`x \in \{0, 1, \ldots, |p|-1\}`
========  ===================================
```

**Probability Mass Function (PMF):**

$$
f(x) = p_x
$$

**Cumulative Distribution Function (CDF):**

$$
F(x \mid p) = \begin{cases}
0 & \text{if } x < 0 \\
\sum_{i=0}^{x} p_i & \text{if } 0 \leq x < |p| \\
1 & \text{if } x \geq |p|
\end{cases}
$$

where \( p \) is the array of probabilities for each category.

```{seealso}
:class: seealso

**Related Distributions:**

- [Bernoulli](bernoulli.md) - The Categorical distribution is a generalization of the Bernoulli distribution to more than two outcomes.
- [Discrete Uniform](discrete_uniform.md) - A special case of the Categorical distribution where all outcomes have equal probability.
```

## References

- [Wikipedia - Categorical Distribution](https://en.wikipedia.org/wiki/Categorical_distribution)





