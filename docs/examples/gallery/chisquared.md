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
# Chi-Squared Distribution

The chi-squared (also chi-square or $\chi^2$) distribution is a fundamental continuous probability distribution in statistics, characterized by its degrees of freedom, $\nu$. This distribution emerges from the sum of the squares of $\nu$ independent standard normal random variables.

The chi-squared distribution is widely used in inferential statistics for many statistical tests, for hypothesis testing and the construction of confidence intervals. It is used in common chi-squared tests for goodness of fit, which compare an observed distribution to a theoretical one, and for testing the independence of two criteria of classification in qualitative data. Additionally, the chi-squared distribution is used for finding confidence intervals to estimate the population standard deviation of a normal distribution based on a sample standard deviation.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Chi-Squared Distribution PDF
---

import matplotlib.pyplot as plt
import arviz as az
from preliz import ChiSquared
az.style.use('arviz-doc')
nus = [1, 3, 9]

for nu in nus:
    ax = ChiSquared(nu).plot_pdf(support=(0, 20))
    ax.set_ylim(0, 0.6)
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Chi-Squared Distribution CDF
---

for nu in nus:
    ChiSquared(nu).plot_cdf(support=(0, 20))
```
## Key properties and parameters:

```{eval-rst}
========  ==========================================
Support   :math:`x \in (0, \infty)`
Mean      :math:`\nu`
Variance  :math:`2\nu`
========  ==========================================
```

**Probability Density Function (PDF):**

$$
f(x|\nu) = \frac{1}{2^{\nu/2}\Gamma(\nu/2)} x^{\nu/2 - 1} e^{-x/2}
$$

where $\Gamma$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

**Cumulative Distribution Function (CDF):**

$$
F(x|\nu) = \frac{1}{\Gamma(\nu/2)} \gamma(\nu/2, x/2)
$$

where $\gamma$ is the [lower incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).


```{seealso}
:class: seealso

**Common Alternatives:**

- [Gamma](gamma.md) - The chi-squared distribution is a special case of the gamma distribution with the shape parameter $\alpha = \nu/2$ and the scale parameter $\beta = 1/2$.

**Related Distributions:**

- [Normal](normal.md) - By definition, the chi-squared distribution is the sum of the squares of $\nu$ independent standard normal random variables.
- [Exponential](exponential.md) - A chi-squared distribution with 2 degrees of freedom is equivalent to an exponential distribution with the rate parameter $\lambda = 1/2$, because the exponential distribution is also a special case of the gamma distribution.
```

## References

- [Wikipedia - Chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution)

