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
# Fréchet Distribution

<audio controls> <source src="../../_static/frechet.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Asymmetric](../../gallery_tags.rst#asymmetric)

The Frechet distribution (also known as the inverse Weibull distribution) is a continuous probability distribution defined on the positive real line. It is one of the three extreme value distributions and is commonly used to model the distribution of maxima, such as maximum river levels, maximum wind speeds, or the largest observed value in a sample. It is parametrized by a shape parameter ($\alpha$) and a scale parameter ($\sigma$).

## Key properties and parameters

```{eval-rst}
========  ==========================================================================
Support   :math:`x \in (0, \infty)`
Mean      :math:`\sigma \Gamma(1 - 1/\alpha) \text{ for } \alpha > 1, \text{ else } \infty`
Variance  :math:`\sigma^2 \left(\Gamma(1 - 2/\alpha) - \Gamma(1 - 1/\alpha)^2\right) \text{ for } \alpha > 2, \text{ else } \infty`
========  ==========================================================================
```

**Parameters:**

- $\alpha$ : (float) Shape parameter, $\alpha > 0$.
- $\sigma$ : (float) Scale parameter, $\sigma > 0$.

### Probability Density Function (PDF)

$$
f(x \mid \alpha, \sigma) =
    \frac{\alpha}{\sigma} \left(\frac{x}{\sigma}\right)^{-1-\alpha}
    e^{-(x/\sigma)^{-\alpha}}
$$

```{jupyter-execute}
:hide-code:

from preliz import Frechet, style
style.use('preliz-doc')
alphas = [1., 2., 3.]
sigmas = [1., 1., 1.]
for alpha, sigma in zip(alphas, sigmas):
    ax = Frechet(alpha, sigma).plot_pdf(support=(0, 10))
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid \alpha, \sigma) = e^{-(x/\sigma)^{-\alpha}}
$$

```{jupyter-execute}
:hide-code:

for alpha, sigma in zip(alphas, sigmas):
    ax = Frechet(alpha, sigma).plot_cdf(support=(0, 10))
```

```{seealso}
:class: seealso

**Common Alternatives:**

- [Weibull](weibull.md) - The Weibull distribution models minima, while the Fréchet distribution models maxima; the reciprocal of a Fréchet variable follows a Weibull distribution..

```

## References

- Wikipedia - [Frechet distribution](https://en.wikipedia.org/wiki/Fr%C3%A9chet_distribution)