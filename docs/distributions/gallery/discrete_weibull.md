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
# Discrete Weibull Distribution

<audio controls> <source src="../../_static/discreteweibull.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Discrete](../../gallery_tags.rst#discrete), [Bounded](../../gallery_tags.rst#bounded)

The Discrete Weibull distribution is a discrete probability distribution used to model the number of trials until an event occurs. It is an analog of the continuous Weibull distribution, adapted for discrete settings. The distribution has two shape parameters: $q$ and $\beta$. The parameter $q$ ($0 < q < 1$) controls the overall probability decay across trials, with smaller values leading to faster decay. The parameter $\beta$ ($\beta > 0$) determines the specific shape of the distribution, influencing how probabilities change as the number of trials increases.

The Discrete Weibull distribution is predominantly used in reliability engineering. It is particularly suitable for modeling failure data measured in discrete units, such as cycles or shocks. This distribution is a versatile tool for analyzing scenarios where event timing is counted in distinct intervals, making it especially valuable in fields that handle discrete data patterns and reliability analysis.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{N}_0`
Mean      :math:`\mu = \sum_{x = 1}^{\infty} q^{x^{\beta}}`
Variance  :math:`2 \sum_{x = 1}^{\infty} x q^{x^{\beta}} - \mu - \mu^2`
========  ===============================================
```

**Parameters:**

- $q$ : (float) Probability of success in each trial, $0 < q < 1$.
- $\beta$ : (float) Shape parameter, $\beta > 0$.

### Probability Mass Function (PMF)

$$
f(x \mid q, \beta) = q^{x^{\beta}} - q^{(x + 1)^{\beta}}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Discrete Weibull Distribution PMF
---

from preliz import DiscreteWeibull, style
style.use('preliz-doc')
qs = [0.1, 0.9, 0.9]
betas = [0.5, 0.5, 2]

for q, beta in zip(qs, betas):
    DiscreteWeibull(q, beta).plot_pdf(support=(0, 10))
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid q, \beta) = 1 - q^{(x + 1)^{\beta}}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Discrete Weibull Distribution CDF
---

for q, beta in zip(qs, betas):
    DiscreteWeibull(q, beta).plot_cdf(support=(0, 10))
```

```{seealso}
:class: seealso

**Common Alternatives:**

- [Geometric](geometric.md) - A special case of the Discrete Weibull distribution with $\beta = 1$.

**Related Distributions:**

- [Weibull](weibull.md) - The continuous analog of the Discrete Weibull distribution.
```

## References

- [Wikipedia - Discrete Weibull Distribution](https://en.wikipedia.org/wiki/Discrete_Weibull_distribution)