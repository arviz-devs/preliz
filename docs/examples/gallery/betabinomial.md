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
# Beta-Binomial Distribution

<audio controls> <source src="../../_static/betabinomial.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Beta-Binomial distribution is a discrete probability distribution derived from the Binomial distribution, with its probability of success in each trial governed by a Beta distribution. This distribution is characterized by three parameters:  $\alpha$, $\beta$, and $n$, where $\alpha$ and $\beta$ are the shape parameters of the Beta distribution and $n$ is the number of trials in the Binomial distribution.

In Bayesian statistics, the Beta-Binomial distribution is commonly used as a prior for the probability of success in a Bernoulli trial. This makes it particularly useful in modeling scenarios with uncertainty in the success probability.


## Probability Mass Function (PMF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Beta-Binomial Distribution PMF
---

from preliz import BetaBinomial, style
style.use('preliz-doc')
alphas = [0.5, 1, 2.3]
betas = [0.5, 1, 2]
n = 10
for a, b in zip(alphas, betas):
    BetaBinomial(a, b, n).plot_pdf()
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Beta-Binomial Distribution CDF
---

for a, b in zip(alphas, betas):
    BetaBinomial(a, b, n).plot_cdf()
```

## Key properties and parameters:

```{eval-rst}
========  =================================================================
Support   :math:`x \in \{0, 1, \ldots, n\}`
Mean      :math:`n \dfrac{\alpha}{\alpha + \beta}`
Variance  :math:`\dfrac{n \alpha \beta (\alpha+\beta+n)}{(\alpha+\beta)^2 (\alpha+\beta+1)}`
========  =================================================================
```

**Probability Mass Function (PMF):**

$$
f(x \mid \alpha, \beta, n) = \binom{n}{x} \frac{B(x + \alpha, n - x + \beta)}{B(\alpha, \beta)}
$$

where $B$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function) and $\binom{n}{x}$ is the [binomial coefficient](https://en.wikipedia.org/wiki/Binomial_coefficient).

**Cumulative Distribution Function (CDF):**

$$
F(x \mid \alpha, \beta, n) = 
\begin{cases} 
0, & x < 0 \\ 
\sum_{k=0}^{x} \binom{n}{k} \frac{B(k + \alpha, n - k + \beta)}{B(\alpha, \beta)}, & 0 \leq x < n \\ 
1, & x \geq n 
\end{cases}
$$

```{seealso}
:class: seealso

**Related Distributions:**

- [Binomial](binomial.md) - The Beta-Binomial distribution is a compound distribution derived from the Binomial distribution.
- [Beta](beta.md) - The Beta-Binomial distribution is a compound distribution parameterized by the Beta distribution.
```

## References

- [Wikipedia - Beta-Binomial](https://en.wikipedia.org/wiki/Beta-binomial_distribution)




