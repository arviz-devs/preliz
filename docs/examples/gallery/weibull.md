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
# Weibull Distribution

<audio controls> <source src="../../_static/weibull.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Weibull distribution is a continuous probability distribution that models the "waiting time" until an event occurs. It has two parameters: the shape parameter $\alpha$ and the scale parameter $\beta$.

This distribution is widely used in fields like engineering, survival analysis, and material science. If $\alpha > 1$, the event becomes more likely as time passes, making it useful for modeling aging or wear-out processes. If $\alpha < 1$, the event is more likely at the beginning and decreases over time.


## Probability Density Function (PDF)

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Weibull Distribution PDF
---

from preliz import Weibull, style
style.use('preliz-doc')
alphas = [1., 2., 5., 5.]
betas = [1., 1., 1., 2.]
for alpha, beta in zip(alphas, betas):
    Weibull(alpha, beta).plot_pdf(support=(0, 5))
```

## Cumulative Distribution Function (CDF)

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Weibull Distribution CDF
---

for alpha, beta in zip(alphas, betas):
    Weibull(alpha, beta).plot_cdf(support=(0, 5))
```

## Key properties and parameters:

```{eval-rst}
========  ==========================================
Support   :math:`x \in [0, \infty)`
Mean      :math:`\beta \Gamma(1 + \frac{1}{\alpha})`
Variance  :math:`\beta^2 \Gamma(1 + \frac{2}{\alpha} - \mu^2/\beta^2)`
========  ====================================================
```

**Probability Density Function (PDF):**

$$
f(x \mid \alpha, \beta) = \frac{\alpha x^{\alpha - 1} \exp(-(\frac{x}{\beta})^{\alpha})}{\beta^\alpha}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x \mid \alpha, \beta) = 1 - \exp(-(\frac{x}{\beta})^{\alpha})
$$

```{seealso}
:class: seealso

**Common Alternatives:**

- [Exponential](exponential.md) - The Exponential distribution is a special case of the Weibull distribution with the shape parameter $\alpha = 1$.

**Related Distributions:**

- [Gumbel](gumbel.md) - If a random variable follows a Weibull distribution, the logarithm of the random variable follows a Gumbel distribution.
```

## References

- [Wikipedia - Weibull Distribution](https://en.wikipedia.org/wiki/Weibull_distribution)

