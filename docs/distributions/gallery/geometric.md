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
# Geometric Distribution

<audio controls> <source src="../../_static/geometric.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Geometric distribution is a discrete probability distribution used to model the number of Bernoulli trials required to achieve the first success. The distribution has a single parameter, $p$ ($0 < p \leq 1$), representing the probability of success in each Bernoulli trial.

The geometric distribution is memoryless, meaning that the probability of success on the next trial is independent of past outcomes. This property makes it useful in scenarios such as modeling the time until the first occurrence of an event in reliability engineering, quality control, and queuing theory. It is particularly effective for analyzing discrete data patterns where each trial is independent and identically distributed.

## Key properties and parameters

```{eval-rst}
========  =============================
Support   :math:`x \in \mathbb{N}_{>0}`
Mean      :math:`\dfrac{1}{p}`
Variance  :math:`\dfrac{1 - p}{p^2}`
========  =============================
```

**Parameters:**

- $p$ : (float) Probability of success in each Bernoulli trial, $0 < p \leq 1$.

### Probability Mass Function (PMF)

$$
f(x \mid p) = p(1-p)^{x-1}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Geometric Distribution PMF
---

from preliz import Geometric, style
style.use('preliz-doc')
for p in [0.1, 0.25, 0.75]:
    Geometric(p).plot_pdf(support=(1,10))
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid p) = 1 - (1-p)^x
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Geometric Distribution CDF
---

for p in [0.1, 0.25, 0.75]:
    Geometric(p).plot_cdf(support=(1,10))
```

```{seealso}
:class: seealso

**Common Alternatives:**

- [Discrete Weibull Distribution](discrete_weibull.md) - The Geometric distribution is a special case of the Discrete Weibull distribution with shape parameter $\beta=1$.

**Related Distributions:**

- [Exponential Distribution](exponential.md) - Continuous analog of the Geometric distribution, used to model the time between events in a Poisson process.
- [Negative Binomial Distribution](negative_binomial.md) - The geometric distribution (on $\{ 0, 1, 2, 3, \dots \}$) is a special case of the negative binomial distribution, with $\text{Geom}(p)=\text{NB}(1, p)$.
```

## References

- [Wikipedia - Geometric Distribution](https://en.wikipedia.org/wiki/Geometric_distribution)