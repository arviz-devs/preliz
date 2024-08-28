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
# Normal Distribution

<audio controls> <source src="../../_static/normal.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The normal distribution, also known as the Gaussian distribution, is a continuous probability distribution characterized by its bell-shaped curve, symmetric around the mean. It is defined by two parameters: the mean ($\mu$) and the standard deviation ($\sigma$). The mean determines the center of the distribution, while the standard deviation controls the spread or width of the distribution.

The normal distribution is often the result of summing many small, independent effects, a concept encapsulated by the Central Limit Theorem. Many biological processes exhibit this property, such as human height, blood pressure, and measurement errors. In such cases, numerous genetic, environmental, and random factors contribute to the observed outcome, each adding a small effect. For example, human height is influenced by multiple genes and environmental factors, which together result in a normal distribution of heights across a population.

Normal priors are commonly chosen in Bayesian analysis because they represent a state of limited prior knowledge or weak information. When we assume a parameter has finite variance but lack strong prior knowledge, the normal distribution becomes a suitable choice. This is due to the normal distribution's maximum entropy property among all distributions with a specified mean and variance, ensuring it introduces the least amount of additional assumptions. 

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Normal Distribution PDF
---

import matplotlib.pyplot as plt

from preliz import Normal, style
style.use('preliz-doc')
mus = [0., 0., -2.]
sigmas = [1, 0.5, 1]
for mu, sigma in zip(mus, sigmas):
    Normal(mu, sigma).plot_pdf()
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Normal Distribution CDF
---

for mu, sigma in zip(mus, sigmas):
    Normal(mu, sigma).plot_cdf()
```


## Key properties and parameters:

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu`
Variance  :math:`\sigma^2`
========  ==========================================
```

**Probability Density Function (PDF):**


$$
f(x \mid \mu, \sigma) =
\frac{1}{\sigma \sqrt{2\pi}}
\exp\left( -\frac{1}{2} \left(\frac{x-\mu}{\sigma}\right)^2 \right)
$$

**Cumulative Distribution Function (CDF):**

$$
F(x \mid \mu, \sigma) =
\frac{1}{2} \left[ 1 + \text{erf} \left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right]
$$

where erf is the [error function](https://en.wikipedia.org/wiki/Error_function).


```{seealso}
:class: seealso

**Common Alternatives:**

- [Student's T](students_t.md) - Commonly used as a robust alternative to the Normal, it has a third parameter controlling the heaviness of the tails.
- [Logistic](logistic.md) - Used as a robust alternative to the Normal, it has heavier tails than the normal distribution.
- [Truncated Normal](truncated_normal.md) - Bounds the Normal distribution within a specified range. Sometimes, it is used for Normal-like parameters when certain values are not permitted.

**Related Distributions:**

- [Half Normal](half_normal.md) - Considers only the positive half of the Normal distribution.
- [Log Normal](log_normal.md) - If a random variable follows a LogNormal distribution, its natural logarithm will be Normally distributed. It is suitable for modelling positive and right-skewed data or parameters.
- [Von Mises](vonmises.md) - Close to a wrapped Normal on the circle. Suitable for data on a circular scale, like angles or time of day.

```

## References

- Wikipedia - [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
