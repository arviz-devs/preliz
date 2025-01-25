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
# Ex-Gaussian Distribution

<audio controls> <source src="../../_static/exgaussian.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Ex-Gaussian distribution (exponentially modified Gaussian distribution or EMG) is a continuous probability distribution that results from the convolution of a normal distribution and an exponential distribution. It is characterized by three parameters: $\mu$, $\sigma$, and $\nu$, which are the mean and standard deviation of the normal component, and the mean of the exponential component, respectively. It has a bell-shaped curve like the normal distribution, but with a positive skew due to the exponential component.

The Ex-Gaussian distribution is commonly used to model reaction times in psychology. It is also used to model the shape of chromatographic peaks, the intermitotic times of cell division, cluster ion beams, and other phenomena.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu + \nu`
Variance  :math:`\sigma^2 + \nu^2`
========  ==========================================
```

**Parameters:**

- $\mu$ : (float) Mean of the normal component.
- $\sigma$ : (float) Standard deviation of the normal component, $\sigma > 0$.
- $\nu$ : (float) Mean of the exponential component, $\nu > 0$.

### Probability Density Function (PDF)

$$
f(x \mid \mu, \sigma, \nu) =
\frac{1}{\nu}\exp\left(\frac{\mu-x}{\nu}+\frac{\sigma^2}{2\nu^2}\right)\Phi\left(\frac{x-\mu}{\sigma}-\frac{\sigma}{\nu}\right)
$$

where $\Phi$ is the [standard normal CDF](normal.md).

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Ex-Gaussian Distribution PDF
---

from preliz import ExGaussian, style
style.use('preliz-doc')
mus = [0., 0., -3.]
sigmas = [1., 3., 1.]
nus = [1., 1., 4.]
for mu, sigma, nu in zip(mus, sigmas, nus):
    ExGaussian(mu, sigma, nu).plot_pdf(support=(-6,9))
```

### Cumulative Distribution Function (CDF)

$$
F(x \mid \mu, \sigma, \nu) = 
 \Phi(x, \mu, \sigma)
- \frac{1}{2} \exp\left( \frac{1}{2\nu} \left(2\mu + \frac{\sigma^2}{\nu} - 2x\right) \right) \left(1 + \operatorname{erf}\left( \frac{x - \left(\mu + \frac{\sigma^2}{\nu}\right)}{\sigma \sqrt{2}} \right)\right)
$$

where $\Phi$ is the CDF of a Gaussian distribution, $\text{erf}$ is the error function

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Ex-Gaussian Distribution CDF
---

for mu, sigma, nu in zip(mus, sigmas, nus):
    ExGaussian(mu, sigma, nu).plot_cdf(support=(-6,9))
```

```{seealso}
:class: seealso

**Related Distributions:**

- [Normal](normal.md) - The Gaussian component of the Ex-Gaussian distribution.
- [Exponential](exponential.md) - The exponential component of the Ex-Gaussian distribution.
```

## References

- [Wikipedia - Ex-Gaussian distribution](https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution)