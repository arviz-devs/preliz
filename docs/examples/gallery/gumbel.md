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
# Gumbel Distribution

<audio controls> <source src="../../_static/gumbel.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Gumbel distribution, also known as the type-I Generalized Extreme Value (GEV) distribution, is a continuous probability distribution that describes the distribution of the maximum (or minimum) of a number of samples of various different random variables, particularly the exponential and normal distributions. It is characterized by two parameters: the location parameter $\mu$ and the scale parameter $\beta$.

The Gumbel distribution is commonly used in the fields of hydrology, meteorology, and environmental science to model extreme events such as floods, earthquakes, and wind speeds.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Gumbel Distribution PDF
---


from preliz import Gumbel, style
style.use('preliz-doc')
mus = [0., 4., -1.]
betas = [1., 2., 4.]
for mu, beta in zip(mus, betas):
    Gumbel(mu, beta).plot_pdf(support=(-10,20))
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Gumbel Distribution CDF
---

for mu, beta in zip(mus, betas):
    Gumbel(mu, beta).plot_cdf(support=(-10,20))
```

## Key properties and parameters:

```{eval-rst}

========  ==========================================
Support   :math:`x \in (-\infty, \infty)`
Mean      :math:`\mu + \beta \gamma`, where :math:`\gamma` is the `Euler-Mascheroni constant <https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant>`_
Variance  :math:`\frac{\pi^2}{6} \beta^2`
========  ==========================================
```

**Probability Density Function (PDF):**

$$
f(x|\mu, \beta) = \frac{1}{\beta} e^{-(z + e^{-z})}
$$

where $z = \frac{x - \mu}{\beta}$.

**Cumulative Distribution Function (CDF):**

$$
F(x|\mu, \beta) = e^{-e^{-z}}
$$

where $z = \frac{x - \mu}{\beta}$.

```{seealso}
:class: seealso

**Related Distributions:**

- [Weibull Distribution](weibull) - If a random variable follows a Weibull distribution, the logarithm of the random variable follows a Gumbel distribution.
- [Exponential Distribution](exponential) - The maximum of a number of exponentially distributed random variables follows a Gumbel distribution and the exponential distribution is a special case of the Weibull distribution.
```


## References

- [Wikipedia - Gumbel Distribution](https://en.wikipedia.org/wiki/Gumbel_distribution)