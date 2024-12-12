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
# Zero-Inflated Poisson Distribution

The Zero-Inflated Poisson (ZIP) distribution is a discrete probability distribution used to model count data characterized by an excess of zeros. It combines two components: a standard Poisson distribution and an additional mechanism that increases the probability of observing zero outcomes.

This distribution is particularly useful for scenarios where the data exhibit more zeros than what the Poisson model alone would predict. For example, in ecological studies, when researchers survey multiple habitats for a particular species, they often encounter many sites with zero observations (typically due to unsuitable habitat conditions) alongside a smaller number of sites where the species is observed in varying counts.

## Key properties and parameters

```{eval-rst}
========  ================================
Support   :math:`x \in \mathbb{N}_0`
Mean      :math:`\psi \mu`
Variance  :math:`\psi \mu (1+(1-\psi) \mu`
========  ================================
```

**Parameters:**

- $\psi$ : (float) Expected proportion of Poisson variates, $0 \leq \psi \leq 1$.
- $\mu$ : (float) The mean rate of events of the Poisson component, $\mu > 0$.

### Probability Mass Function (PMF)

$$
f(x \mid \psi, \mu) = \left\{ \begin{array}{l}
    (1-\psi) + \psi e^{-\mu}, \text{if } x = 0 \\
    \psi \frac{e^{-\mu}\mu^x}{x!}, \text{if } x=1,2,3,\ldots
    \end{array} \right.
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Zero-Inflated Poisson Distribution PMF
---

from preliz import ZeroInflatedPoisson, style
style.use('preliz-doc')
psis = [0.7, 0.4]
mus = [8, 4]
for psi, mu in zip(psis, mus):
    ZeroInflatedPoisson(psi, mu).plot_pdf()
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid \psi, \mu) = \left\{ \begin{array}{l}
    (1-\psi) + \psi e^{-\mu}, \text{if } x = 0 \\
    (1-\psi) + \psi \sum_{k=0}^x \frac{e^{-\mu}\mu^k}{k!}, \text{if } x=1,2,3,\ldots
    \end{array} \right.
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Zero-Inflated Poisson Distribution CDF
---

for psi, mu in zip(psis, mus):
    ZeroInflatedPoisson(psi, mu).plot_cdf()
```

```{seealso}

**Common Alternatives:**

- [Zero-Inflated Negative Binomial Distribution](zeroinflatednegativebinomial.md) - A distribution for overdispersed count data with excess zeros.
- [Zero-Inflated Binomial Distribution](zeroinflatedbinomial.md) - A related distribution for count data with excess zeros.

**Related Distributions:**

- [Poisson Distribution](poisson.md) - A standard distribution for modeling count data. 
```

## References

- [Wikipedia - Zero-Inflated Models](https://en.wikipedia.org/wiki/Zero-inflated_model)
- [Wikipedia - Poisson Distribution](https://en.wikipedia.org/wiki/Poisson_distribution)