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
# Poisson Distribution

<audio controls> <source src="../../_static/poisson.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Discrete](../../gallery_tags.rst#discrete), [Non-Negative](../../gallery_tags.rst#non-negative)

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time (or space) if these events occur with a known constant mean rate and independently of the time since the last event.

## Key properties and parameters

```{eval-rst}
========  ==========================
Support   :math:`x \in \mathbb{N}_0`
Mean      :math:`\mu`
Variance  :math:`\mu`
========  ==========================
```

**Parameters:**

- $\mu$ : (float) The mean rate of events, $\mu > 0$.

### Probability Density Function (PDF)

$$
f(x \mid \mu) = \frac{e^{-\mu}\mu^x}{x!}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Poisson Distribution PDF
---


from preliz import Poisson, style
style.use('preliz-doc')
for mu in [0.5, 3, 8]:
    Poisson(mu).plot_pdf();
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid \mu) = \frac{\Gamma(x + 1, \mu)}{x!}
$$

where $\Gamma(x + 1, \mu)$ is the [upper incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Poisson Distribution CDF
---

for mu in [0.5, 3, 8]:
    Poisson(mu).plot_cdf();
```

```{seealso}
:class: seealso

**Common Alternatives:**

- [NegativeBinomial](negativebinomial.md) - The Negative Binomial is often used as an alternative the Poisson when the variance is greater than the mean (overdispersed data).
- [ZeroInflatedPoisson](zeroinflatedpoisson.md) - The Zero-Inflated Poisson is used when there is an excess of zero counts in the data.
- [HurdlePoisson](hurdle.md) - The Hurdle Poisson is used when there is an excess of zero counts in the data.

**Related Distributions:**

- [Binomial](binomial.md) - The Poisson distribution can be derived as a limiting case to the binomial distribution as the number of trials goes to infinity and the expected number of successes remains fixed. See [law of rare events](https://en.wikipedia.org/wiki/Poisson_distribution#law_of_rare_events). 
- [Normal](normal.md) - For sufficiently large values of $\mu$, the normal distribution with mean $\mu$ and standard deviation \sqrt{\mu} can be a good approximation to the Poisson. 
```

## References

- [Wikipedia - Poisson](https://en.wikipedia.org/wiki/Poisson_distribution)