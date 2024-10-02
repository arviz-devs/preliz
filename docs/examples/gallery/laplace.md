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
# Laplace Distribution

<audio controls> <source src="../../_static/laplace.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Laplace or Laplacian distribution is a continuous probability distribution named after Pierre-Simon Laplace. It is also known as the double exponential distribution due to its shape: it consists of two exponential distributions of the same scale back to back about $x = \mu$. The difference between two independent identically distributed (i.i.d.) exponential random variables follows a Laplace distribution.

The Laplace distribution is characterized by two parameters: the location parameter $\mu$ and the scale parameter $b$.

The Laplace distribution is widely used due to its ability to model sharp peaks and heavy tails. In signal processing, it models coefficients in speech recognition and image compression. In regression analysis, the Lasso regression can be thought of as a Bayesian regression with a Laplacian prior for the coefficients, altought sparsity is better achieved with priors such as horseshoe instead of Laplace. It is also used in hydrology, finance, and other fields to model extreme events and outliers.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
import matplotlib.pyplot as plt

from preliz import Laplace, style
style.use('preliz-doc')

mus = [0., 0., 0., -5.]
bs = [1., 2., 4., 4.]
for mu, b in zip(mus, bs):
    Laplace(mu, b).plot_pdf(support=(-10,10))
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
import matplotlib.pyplot as plt

for mu, b in zip(mus, bs):
    Laplace(mu, b).plot_cdf(support=(-10,10))
```

## Key properties and parameters:

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu`
Variance  :math:`2b^2`
========  ==========================================
```

**Probability Density Function (PDF):**

$$
f(x|\mu, b) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)
$$

**Cumulative Distribution Function (CDF):**

$$
F(x|\mu, b) = \begin{cases}
\frac{1}{2} \exp\left(\frac{x - \mu}{b}\right) & \text{if } x < \mu \\
1 - \frac{1}{2} \exp\left(-\frac{x - \mu}{b}\right) & \text{if } x \geq \mu
\end{cases}
$$

```{seealso}
:class: seealso

**Common Alternatives:**

- [Asymmetric Laplace](asymmetric_laplace.md) - A generalization of the Laplace distribution with asymmetric tails.

**Related Distributions:**

- [Exponential](exponential.md) - The Laplace distribution is the difference between two i.i.d. exponential random variables.
```

## References

- [Wikipedia - Laplace Distribution](https://en.wikipedia.org/wiki/Laplace_distribution)



