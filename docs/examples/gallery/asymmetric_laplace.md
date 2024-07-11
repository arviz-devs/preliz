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
# Asymmetric Laplace Distribution

The Asymmetric Laplace distribution (ALD) is a continuous probability distribution. It is a generalization of the Laplace distribution. Just as the Laplace distribution consists of two exponential distributions of equal scale back-to-back about $x = \mu$, the ALD consists of two exponential distributions of unequal scale back to back about $x = \mu$, adjusted to assure continuity and normalization. 

The difference of two variates exponentially distributed with different means and rate parameters will be distributed according to the ALD. When the two rate parameters are equal, the difference will be distributed according to the Laplace distribution.

The Asymmetric Laplace distribution, with parameters  $\mu$, $b$ and $q$, is commonly used for performing quantile regression in a Bayesian inference context, with $q$ indicating the desired quantile. 

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: AsymmetricLaplace Distribution PDF
---

import arviz as az
from preliz import AsymmetricLaplace
az.style.use('arviz-doc')
kappas = [1., 2., .5]
mus = [0., 0., 3.]
bs = [1., 1., 1.]
for kappa, mu, b in zip(kappas, mus, bs):
    AsymmetricLaplace(kappa, mu, b).plot_pdf(support=(-10, 10))
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: AsymmetricLaplace Distribution CDF
---

for kappa, mu, b in zip(kappas, mus, bs):
    AsymmetricLaplace(kappa, mu, b).plot_cdf(support=(-10, 10))
```


## Key properties and parameters:

```{eval-rst}
========  =========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu-\frac{\\\kappa-1/\kappa}b`
Variance  :math:`\frac{1+\kappa^{4}}{b^2\kappa^2 }`
========  =========================================
```

**Probability Density Function (PDF):**


$$
{f(x \mid b,\kappa,\mu) =
    \left({\frac{b}{\kappa + 1/\kappa}}\right)\, e^{-(x-\mu) b \,s\kappa ^{s}}}
$$

where s=sgn(x-m), and [sgn](https://en.wikipedia.org/wiki/Sign_function) is the sign function.

**Cumulative Distribution Function (CDF):**

$$
F(x \mid b,\kappa,\mu)  = 
    \begin{cases}
      \frac{\kappa^2}{1+\kappa^2}\exp(b \kappa(x-\mu)) & \text{if } x \leq \mu \\
     1-\frac{1}{1+\kappa^2} \exp(-b \kappa(x-\mu))  & \text{if } x > \mu
    \end{cases}
$$


```{seealso}
:class: seealso

**Related Distributions:**
- [Laplace](laplace_distribution.md) - The Asymmetric Laplace distribution is a generalization of the Laplace distribution.
```

## References

- Wikipedia - [AsymmetricLaplace distribution](https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution)