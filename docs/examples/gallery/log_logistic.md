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
# Log-Logistic Distribution

<audio controls> <source src="../../_static/loglogistic.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The log-logistic distribution, also known as the Fisk distribution, is a continuous probability distribution that is used to model non-negative random variables. It is characterized by two parameters: a scale parameter ($\alpha$), which is also the median of the distribution, and a shape parameter ($\beta$). The dispersion of the distribution decreases as $\beta$ increases.

The log-logistic distribution is often used in survival analysis as a parametric model for events whose rate increases initially and decreases later, as, for example, mortality rate from cancer following diagnosis or treatment. It has been used in hydrology for modelling stream flow rates and precipitation. It is also used in reliability engineering to model the lifetime of components and systems.


## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Log-Logistic Distribution PDF
---

import matplotlib.pyplot as plt

from preliz import LogLogistic, style
style.use('preliz-doc')
alphas = [1, 1, 1, 1, 2]
betas = [1, 2, 4, 8, 8]
for alpha, beta in zip(alphas, betas):
    LogLogistic(alpha, beta).plot_pdf(support=(0, 6))
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Log-Logistic Distribution CDF
---

for alpha, beta in zip(alphas, betas):
    LogLogistic(alpha, beta).plot_cdf(support=(0, 6))
```

## Key properties and parameters:

```{eval-rst}
========  ==========================================================================
Support   :math:`x \in [0, \infty)`
Mean      :math:`{\alpha\,\pi/\beta \over \sin(\pi/\beta)}` if :math:`\beta>1`, else undefined
Variance  :math:`\alpha^2 \left(2b / \sin 2b -b^2 / \sin^2 b \right), \quad \beta>2`
========  ==========================================================================
```

**Probability Density Function (PDF):**

$$
f(x|\alpha, \beta) =   \frac{ (\beta/\alpha)(x/\alpha)^{\beta-1}}{\left( 1+(x/\alpha)^{\beta} \right)^2}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x|\alpha, \beta) = \frac{1}{1 + (x/\alpha)^{-\beta}}
$$

```{seealso}
:class: seealso

**Common Alternatives:**

- [Log-Normal Distribution](log_normal.md) - The log-logistic distribution is similar to the log-normal distribution, but with heavier tails.

**Related Distributions:**

- [Logistic Distribution](logistic.md) - If a random variable $X$ is distributed as a log-logistic distribution, then its logarithm $\log(X)$ follows a logistic distribution.
```

## References

- [Wikipedia - Log-logistic distribution](https://en.wikipedia.org/wiki/Log-logistic_distribution)