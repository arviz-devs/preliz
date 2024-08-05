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
# Log-Normal Distribution

The log-normal distribution is a continuous probability distribution of a random variable whose logarithm is normally distributed. Thus, if a random variable $X$ follows a log-normal distribution, then $ Y = \log(X)$ is normally distributed. It has a right-skewed shape and is defined for positive values of $x$. It is is characterized by two parameters: $\mu$ and $\sigma$, which are the mean and standard deviation of the log-transformed variable, respectively, not the original variable.

The log-normal distribution is commonly used to model variables that are positive and result from the product of many small independent factors (instead of the sum of factors, as in the normal distribution). This property makes it a widespread choice for modeling quantities in many fields of knowledge, including biology, engineering, medicine, finance and others. For example, In hydrology, the log-normal distribution is used to model the distribution of annual maximum rainfall, river discharge, and other hydrological variables.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Log-Normal Distribution PDF
---

import matplotlib.pyplot as plt
from preliz import style, LogNormal
style.use('preliz-doc')
mus = [0., 0., 0.]
sigmas = [0.25, 0.5, 1.]
for mu, sigma in zip(mus, sigmas):
    LogNormal(mu, sigma).plot_pdf(support=(0, 5))
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Log-Normal Distribution CDF
---

for mu, sigma in zip(mus, sigmas):
    LogNormal(mu, sigma).plot_cdf(support=(0, 5))
```

## Key properties and parameters:

```{eval-rst}
========  ==========================================
Support   :math:`x \in (0, \infty)`
Mean      :math:`e^{\mu + \frac{\sigma^2}{2}}`
Variance  :math:`(e^{\sigma^2} - 1)e^{2\mu + \sigma^2}`
========  ==========================================
```

**Probability Density Function (PDF):**

$$
f(x \mid \mu, \sigma) =
\frac{1}{x \sigma \sqrt{2\pi}}
\exp\left( -\frac{1}{2} \left(\frac{\log(x)-\mu}{\sigma}\right)^2 \right)
$$

**Cumulative Distribution Function (CDF):**

$$
F(x \mid \mu, \sigma) = \frac{1}{2} + \frac{1}{2} \text{erf} \left( \frac{\log(x) - \mu}{\sigma \sqrt{2}} \right)
$$


```{seealso}
:class: seealso

**Common Alternatives:**

- [Normal](normal.md) - The log-normal distribution is directly related to the normal distribution since if a variable is log-normally distributed, its logarithm follows a normal distribution. This relationship is crucial for understanding the log-normal's properties and applications.

**Related Distributions:**

- [Exponential](exponential.md) - It's a simpler model for positive, skewed data but lacks the flexibility of the log-normal distribution in modeling a wide range of shapes.
- [Gamma](gamma.md) - Used for modeling positively skewed data, similar to the log-normal distribution, but with different skewness and kurtosis properties.
- [Pareto](pareto.md) - Also models positive, skewed data, focusing on the "long tail" of distribution which is useful in economics and finance for modeling wealth distribution or significant rare events.
```

## References

1. Wikipedia - [Log-normal distribution](https://en.wikipedia.org/wiki/Log-normal_distribution)




