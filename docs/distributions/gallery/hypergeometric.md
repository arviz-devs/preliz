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
# Hypergeometric Distribution

<audio controls> <source src="../../_static/hypergeometric.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Discrete](../../gallery_tags.rst#discrete), [Bounded](../../gallery_tags.rst#bounded)

The Hypergeometric distribution is a discrete probability distribution used to model the number of successes in a sequence of $n$ draws, taken whithout replacement, from a finite population of size $N$ containing $K$ successful individuals. The distribution is defined by three parameters: $N$, $K$, and $n$.

This distribution is particularly useful for sampling without replacement, where the probability of success changes with each draw. Common applications include calculating probabilities in games of chance like lottery draws or poker, auditing elections to detect irregularities, and assessing over-representation in statistical tests such as Fisher's exact test.

## Key properties and parameters

```{eval-rst}
========  =============================
Support   :math:`x \in \left[\max(0, n - N + k), \min(k, n)\right]`
Mean      :math:`\dfrac{nk}{N}`
Variance  :math:`\dfrac{(N-n)nk(N-k)}{(N-1)N^2}`
========  =============================
```

**Parameters:**

- $N$ : (int) Population size. $N > 0$.
- $K$ : (int) Number of successful individuals in the population. $0 \leq K \leq N$.
- $n$ : (int) Number of samples drawn from the population. $0 \leq n \leq N$.

### Probability Mass Function (PMF)

$$
f(x \mid N, K, n) = \frac{\binom{k}{x} \binom{N-K}{n-x}}{\binom{N}{n}}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Hypergeometric Distribution PMF
---

from preliz import HyperGeometric, style
style.use('preliz-doc')
N = 50
k = 10
for n in [20, 25]:
    HyperGeometric(N, k, n).plot_pdf(support=(1,15))
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid N, K, n) = 1 - \frac{\binom{n}{k+1} \binom{N-n}{K-k-1}}{\binom{N}{K}} \, {}_3F_2 \left[ 
\begin{array}{c}
1, \, k+1-K, \, k+1-n \\
k+2, \, N+k+2-K-n
\end{array}
; 1 \right]
$$

where ${}_pF_q$ is the [generalized hypergeometric function](https://en.wikipedia.org/wiki/Generalized_hypergeometric_function).

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Hypergeometric Distribution CDF
---

for n in [20, 25]:
    HyperGeometric(N, k, n).plot_cdf(support=(1,15))
```

```{seealso}
:class: seealso

**Related Distributions:**

- [Binomial Distribution](binomial.md) - The Hypergeometric distribution converges to the Binomial distribution as the population size $N$ approaches infinity.
```

## References

- [Wikipedia - Hypergeometric Distribution](https://en.wikipedia.org/wiki/Hypergeometric_distribution)

