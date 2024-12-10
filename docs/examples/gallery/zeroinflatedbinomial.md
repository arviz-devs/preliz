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
# Zero-Inflated Binomial Distribution

The Zero-Inflated Binomial (ZIB) distribution is a discrete probability distribution used to model count data characterized by an excess of zeros. It combines two components: a standard Binomial distribution and an additional mechanism that increases the probability of observing zero outcomes.

This distribution is particularly useful for scenarios where the data exhibit more zeros than what the Binomial model alone would predict. For example, in a study of the number of doctor visits in a year, many individuals might not visit a doctor at all, resulting in a higher frequency of zero counts than expected under a traditional Binomial model.

## Key properties and parameters

```{eval-rst}
========  ==========================
Support   :math:`x \in \mathbb{N}_0`
Mean      :math:`\psi n p`
Variance  :math:`\psi n p (1 - p) + n^2 p^2 (\psi - \psi^2)`
========  ==========================
```

**Parameters:**

- $\psi$ : (float) Expected proportion of Binomial variates, $0 \leq \psi \leq 1$.
- $n$ : (int) Number of Bernoulli trials, $n \geq 0$.
- $p$ : (float) Probability of success in each trial, $0 \leq p \leq 1$.

### Probability Mass Function (PMF)

$$
f(x \mid \psi, n, p) = \left\{ \begin{array}{l}
    (1-\psi) + \psi (1-p)^{n}, \text{if } x = 0 \\
    \psi {n \choose x} p^x (1-p)^{n-x}, \text{if } x=1,2,3,\ldots,n
    \end{array} \right.
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Zero-Inflated Binomial Distribution PMF
---

from preliz import ZeroInflatedBinomial, style
style.use('preliz-doc')
ns = [10, 20]
ps = [0.5, 0.7]
psis = [0.7, 0.4]
for n, p, psi in zip(ns, ps, psis):
    ZeroInflatedBinomial(psi, n, p).plot_pdf(support=(0,25))
```

### Cumulative Distribution Function (CDF)

$$
F(k \mid \psi, n, p) = \left\{ \begin{array}{l}
    (1-\psi) + \psi (1-p)^{n}, \text{if } k = 0 \\
    (1-\psi) + \psi (1-p)^{n} + \sum_{x=1}^{k} {n \choose x} p^x (1-p)^{n-x}, \text{if } k=1,2,3,\ldots,n
    \end{array} \right.
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Zero-Inflated Binomial Distribution CDF
---

for n, p, psi in zip(ns, ps, psis):
    ZeroInflatedBinomial(psi, n, p).plot_cdf(support=(0,25))
```

```{seealso}

**Common Alternatives:**

- [Binomial Distribution](binomial.md): Models the number of successes in a fixed number of independent Bernoulli trials, each with the same probability of success.
- [Zero-Inflated Poisson Distribution](zeroinflatedpoisson.md): Extends the Poisson distribution by adding a mechanism to account for excess zeros, often observed in count data.

**Related Distributions:**

- [Zero-Inflated Negative Binomial Distribution](zeroinflatednegativebinomial.md): Combines the Negative Binomial distribution with a zero-inflation component, modeling scenarios with overdispersion and an excess of zeros.
```

## References

- [Wikipedia - Zero-Inflated Models](https://en.wikipedia.org/wiki/Zero-inflated_model)
- [Wikipedia - Binomial Distribution](https://en.wikipedia.org/wiki/Binomial_distribution)