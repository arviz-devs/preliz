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
# Logistic Distribution

<audio controls> <source src="../../_static/logistic.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Symmetric](../../gallery_tags.rst#symmetric), [Unbounded](../../gallery_tags.rst#unbounded), [Light-tailed](../../gallery_tags.rst#light-tailed)

The logistic distribution is a continuous probability distribution with a shape that resembles the normal distribution but with heavier tails. Thus, it is sometimes used as a replacement for the normal when heavier tails are needed. It is defined by two parameters: the mean ($\mu$) and the scale parameter ($s$). The mean determines the center of the distribution, while the scale parameter controls the width.

Its cumulative distribution function is the [logistic function](https://en.wikipedia.org/wiki/Logistic_function), which is characterized by an S-shaped curve (sigmoid curve). It is particularly useful in modeling growth processes, such as population growth, where the rate of growth decreases as the population reaches its carrying capacity. 

A logistic regression model is typically characterized by a Bernoulli distribution for the likelihood and the logistic function as the inverse link function. However, logistic regression can also be [described](https://en.wikipedia.org/wiki/Logistic_distribution#Logistic_regression) as a latent variable model where the error term follows a logistic distribution.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu`
Variance  :math:`\frac{\pi^2}{3}s^2`
========  ==========================================
```

**Parameters:**

- $\mu$ : (float) Mean of the distribution.
- $s$ : (float) Scale parameter, $s > 0$.

### Probability Density Function (PDF)

$$
f(x \mid \mu, s) = 
\frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Logistic Distribution PDF
---

import matplotlib.pyplot as plt

from preliz import Logistic, style
style.use('preliz-doc')
mus = [0., 0., -2.]
ss = [1., 2., .4]
for mu, s in zip(mus, ss):
    Logistic(mu, s).plot_pdf(support=(-5,5))
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid \mu, s) = \frac{1}{1 + e^{-(x - \mu) / s}}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Logistic Distribution CDF
---

for mu, s in zip(mus, ss):
    Logistic(mu, s).plot_cdf(support=(-5,5))
```

```{seealso}
:class: seealso

**Common Alternatives:**

- [Normal](normal.md) - Often used as an alternative to the logistic distribution when the tails are not of primary concern.
- [Cauchy](cauchy.md) - Has much heavier tails than the logistic distribution, making it a robust alternative when outliers are a concern.
- [Student's t](students_t.md) - A generalization of the normal distribution with heavier tails.

**Related Distributions:**

- [Log-logistic](log_logistic.md) - If a random variable is distributed as a logistic, then its exponential is distributed as a log-logistic distribution.
```

## References

- [Wikipedia - Logistic](https://en.wikipedia.org/wiki/Logistic_distribution)