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
# Binomial Distribution

<audio controls> <source src="../../_static/binomial.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Binomial distribution is a discrete probability distribution that describes the number of successes in a fixed number $n$ of independent Bernoulli trials (yes/no experiments), each with the same probability of success $p$. 

## Key properties and parameters

```{eval-rst}
========  =================================================================
Support   :math:`x \in \{0, 1, \ldots, n\}`
Mean      :math:`n p`
Variance  :math:`n p (1-p)`
========  =================================================================
```

**Parameters:**

- $n$ : (int) Number of Bernoulli trials, $n \geq 0$.
- $p$ : (float) Probability of success in each trial, $0 \leq p \leq 1$.

### Probability Mass Function (PMF)

$$
f(x \mid n, p) = \binom{n}{x} p^x (1-p)^{n-x}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Binomial Distribution PMF
---

from preliz import Binomial, style
style.use('preliz-doc')
ns = [5, 10, 10]
ps = [0.5, 0.5, 0.7]
for n, p in zip(ns, ps):
    Binomial(n, p).plot_pdf()
```

### Cumulative Distribution Function (CDF)

$$
F(k \mid n, p) = I_{1 - p}(n - \lfloor x \rfloor, \lfloor x \rfloor + 1)
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Binomial Distribution CDF
---

for n, p in zip(ns, ps):
    Binomial(n, p).plot_cdf()
```

where $I_{1 - p}(a, b)$ is the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function).

```{seealso}
:class: seealso

**Common Alternatives:**

- [Bernoulli Distribution](bernoulli.md) - For a single trial, i.e., $n=1$, the Binomial distribution reduces to the Bernoulli distribution.

**Related Distributions:**

- [Beta-Binomial Distribution](betabinomial.md) - Generalization of the Binomial distribution with a Beta distribution for the probability of success.
- [Hypergeometric Distribution](hypergeometric.md) - Distribution of the number of successes in a sample drawn without replacement.
```

## References

- [Wikipedia - Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)