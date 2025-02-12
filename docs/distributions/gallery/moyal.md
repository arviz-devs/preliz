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
# Moyal Distribution

<audio controls> <source src="../../_static/moyal.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Asymmetric](../../gallery_tags.rst#asymmetric), [Unbounded](../../gallery_tags.rst#unbounded), [Light-tailed](../../gallery_tags.rst#light-tailed)

The Moyal distribution is a continuous probability distribution that was proposed by the physicist J. E. Moyal in 1955 as an approximation to the [Landau distribution](https://en.wikipedia.org/wiki/Landau_distribution). The Moyal distribution is characterized by two parameters: the location parameter $\mu$ and the scale parameter $\sigma$. 

The Moyal distribution is used in high-energy physics to model the energy loss, and the number of ion pairs produced, by ionization for fast charged particles.

## Key properties and parameters

```{eval-rst}
========  ==============================================================
Support   :math:`x \in (-\infty, \infty)`
Mean      :math:`\mu + \sigma\left(\gamma + \log 2\right)`, where
          :math:`\gamma` is the `Euler-Mascheroni constant <https://en.wikipedia.org/wiki/Euler%E2%80%93Mascheroni_constant>`_
Variance  :math:`\frac{\pi^{2}}{2}\sigma^{2}`
========  ==============================================================
```

**Parameters:**

- $\mu$ (loc): The location parameter.
- $\sigma$ (scale): The scale parameter.

### Probability Density Function (PDF)

$$
f(x|\mu, \sigma) = \frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{1}{2}\left(z + e^{-z}\right)}
$$

where $z = \frac{x - \mu}{\sigma}$.

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
import matplotlib.pyplot as plt

from preliz import Moyal, style
style.use('preliz-doc')
mus = [-1., 0., 4.]
sigmas = [2., 1., 4.]
for mu, sigma in zip(mus, sigmas):
    Moyal(mu, sigma).plot_pdf(support=(-10,20))
```

### Cumulative Distribution Function (CDF)

$$
F(x; \mu, \sigma) = 1 - \text{erf}\left( \frac{e^{-z}}{\sqrt{2}} \right)
$$

where [erf](https://en.wikipedia.org/wiki/Error_function) is the error function and $z = \frac{x - \mu}{\sigma}$.

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
for mu, sigma in zip(mus, sigmas):
    Moyal(mu, sigma).plot_cdf(support=(-10,20))
```

```{seealso}
:class: seealso

**Related Distributions:**

- [Gamma](gamma.md) - The Moyal distribution is a transformation of the Gamma distribution.
- [LogNormal](log_normal.md) - The LogNormal distribution is also used sometimes as an approximation to the Landau distribution. It is suitable for modeling positive and right-skewed data.
```

## References

- [SciPy Moyal Distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.moyal.html)
- [Wolfram Moyal Distribution](https://reference.wolfram.com/language/ref/MoyalDistribution.html)