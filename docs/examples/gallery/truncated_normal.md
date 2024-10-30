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
# Truncated Normal Distribution

<audio controls> <source src="../../_static/truncatednormal.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The truncated normal distribution is a continuous probability distribution that is a normal distribution restricted to a specific range. It is defined by four parameters: the mean ($\mu$), the standard deviation ($\sigma$), and the lower and upper bounds of the range. 

Truncated normal distributions are commonly used in cases where the observed data is known to fall within a certain range due to physical constraints or measurement limitations.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Truncated Normal Distribution PDF
---

from preliz import TruncatedNormal, style
style.use('preliz-doc')
mus = [0.,  0., 0.]
sigmas = [3.,5.,7.]
lowers = [-3, -5, -5]
uppers = [7, 5, 4]
for mu, sigma, lower, upper in zip(mus, sigmas,lowers,uppers):
    TruncatedNormal(mu, sigma, lower, upper).plot_pdf(support=(-10,10))
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Truncated Normal Distribution CDF
---

for mu, sigma, lower, upper in zip(mus, sigmas,lowers,uppers):
    TruncatedNormal(mu, sigma, lower, upper).plot_cdf(support=(-10,10))
```

## Key properties and parameters:

```{eval-rst}

========  ==========================================
Support   :math:`x \in [lower, upper]`
Mean      :math:`\mu +{\frac {\phi (\alpha )-\phi (\beta )}{Z}}\sigma`
Variance  :math:`\sigma^2 \left[1+\frac{\alpha \phi(\alpha)-\beta \phi(\beta)}{Z}-\left(\frac{\phi(\alpha)-\phi(\beta)}{Z}\right)^2\right]`
========  ==========================================
```
where:

- $\phi$ is the standard normal PDF
- $\alpha = \frac{lower-\mu}{\sigma}$
- $\beta = \frac{upper-\mu}{\sigma}$
- $Z = \Phi(\beta) - \Phi(\alpha)$
- $\Phi$ is the standard normal CDF

**Probability Density Function (PDF):**

$$
\begin{cases}
    0 & \text{for } x < \text{lower}, \\
    \frac{\phi\left(\frac{x-\mu}{\sigma}\right)}{\sigma(Z)} & \text{for } \text{lower} <= x <= \text{upper}, \\
    0 & \text{for } x > \text{upper},
\end{cases}
$$

where $\phi$ is the standard normal PDF, and $Z = \Phi(\beta) - \Phi(\alpha)$.

**Cumulative Distribution Function (CDF):**

$$
\begin{cases}
    0 & \text{for } x < \text{lower}, \\
    \frac{\Phi\left(\frac{x-\mu}{\sigma}\right) - \Phi(\alpha)}{Z} & \text{for } \text{lower} <= x <= \text{upper}, \\
    1 & \text{for } x > \text{upper},
\end{cases}
$$

where $\Phi$ is the standard normal CDF, and $Z = \Phi(\beta) - \Phi(\alpha)$.

```{seealso}
:class: seealso

**Related Distributions:**

- [Normal Distribution](normal.md) - The normal distribution is an unbounded version of the truncated normal distribution.
- [Truncated Distribution](truncated.md) - A modifier of a base distribution that restricts the values to a specific range.
```

## References

- Wikipedia - [Truncated Normal Distribution](https://en.wikipedia.org/wiki/Truncated_normal_distribution)
