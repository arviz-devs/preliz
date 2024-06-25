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

The logistic distribution is a continuous probability distribution with a shape that resembles the normal distribution, but with heavier tails. It is defined by two parameters: the mean ($\mu$) and the scale parameter ($s$). The mean determines the center of the distribution, while the scale parameter controls the steepness of the curve.

Its cumulative distribution function is the [logistic function](https://en.wikipedia.org/wiki/Logistic_function), which is characterized by an S-shaped curve (sigmoid curve). It is particularly useful in modeling growth processes, such as population growth, where the rate of growth decreases as the population reaches its carrying capacity. 

In logistic regression (whether frequentist or Bayesian flavor), the logistic distribution is used to model the probability of a binary outcome based on one or more predictor variables. The logistic function maps any real value into the range [0, 1], making it suitable for binary classification tasks. 

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Logistic Distribution PDF
---

import matplotlib.pyplot as plt
import arviz as az
from preliz import Logistic
az.style.use('arviz-doc')
mus = [0., 0., -2.]
ss = [1., 2., .4]
for mu, s in zip(mus, ss):
    Logistic(mu, s).plot_pdf(support=(-5,5))
```

## Cumulative Distribution Function (CDF):

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


## Key properties and parameters:

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu`
Variance  :math:`\frac{\pi^2}{3}s^2`
========  ==========================================
```

**Probability Density Function (PDF):**


$$
f(x \mid \mu, s) = 
\frac{e^{-(x-\mu)/s}}{s(1+e^{-(x-\mu)/s})^2}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x \mid \mu, s) = \frac{1}{1 + e^{-(x - \mu) / s}}
$$

```{seealso}
:class: seealso

**Common Alternatives:**

- [Normal Distribution](normal_distribution.md) - Often used as an alternative to the logistic distribution when the tails are not of primary concern.
- [Cauchy Distribution](cauchy_distribution.md) - Has much heavier tails than the logistic distribution, making it a robust alternative when outliers are a concern.
- [Student's t Distribution](students_t_distribution.md) - A generalization of the normal distribution with heavier tails.

**Related Distributions:**

- [Log-logistic Distribution](log_logistic_distribution.md) - The probability distribution of a random variable whose logarithm has a logistic distribution.
```

## References

- [Wikipedia - Logistic Distribution](https://en.wikipedia.org/wiki/Logistic_distribution)