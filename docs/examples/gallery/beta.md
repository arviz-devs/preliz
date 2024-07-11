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
# Beta Distribution

The Beta distribution is a continuous probability distribution bounded between 0 and 1. It is usually defined by two positive shape parameters: ($\alpha$) and ($\beta$). But other parametrization like mean ($\mu$) and concentration ($\nu$) are also common.

The Beta distribution can adopt a wide range of "shapes" including uniform, U-shape, normal-like, exponential-like, and many others, always restricted to the unit interval. This flexibility makes it a versatile choice for modeling random variables that represent proportions, probabilities, or rates. For example, the Beta distribution is commonly used to model the uncertainty of the true proportion of successes in a series of Bernoulli trials, where $\alpha$ and $\beta$ represent the number of successes and failures, respectively.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Beta Distribution PDF
---

import matplotlib.pyplot as plt
import arviz as az
from preliz import Beta
az.style.use('arviz-doc')
alphas = [.5, 5., 2.]
betas = [.5, 5., 5.]
for alpha, beta in zip(alphas, betas):
    ax = Beta(alpha, beta).plot_pdf()
ax.set_ylim(0, 5);
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Beta Distribution CDF
---

for mu, sigma in zip(alphas, betas):
    Beta(mu, sigma).plot_cdf()
```


## Key properties and parameters:

```{eval-rst}
========  ==============================================================
Support   :math:`x \in (0, 1)`
Mean      :math:`\dfrac{\alpha}{\alpha + \beta}`
Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
========  ==============================================================
```

**Probability Density Function (PDF):**


$$
f(x \mid \alpha, \beta) =
    \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}
$$

where $B(\alpha,\beta)$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function) 

**Cumulative Distribution Function (CDF):**

$$
F(x \mid \alpha,\beta) = \frac{B(x;\alpha,\beta)}{B(\alpha,\beta)} = I_x(\alpha,\beta)
$$


where $B(x;\alpha,\beta)$ is the [Incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function) and $I_x(\alpha,\beta)$ is the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function).



```{seealso}
:class: seealso

**Common Alternatives:**

- [Kumaraswamy Distribution](Kumaraswamy_distribution.md) -  It is similar to the Beta distribution, but with closed form expression for its probability density function, cumulative distribution function and quantile function.

**Related Distributions:**
- [Beta Scaled](beta_scaled_distribution.md) - A Beta distribution defined on an arbitrary range.
- [Uniform Distribution](uniform_distribution.md) - The Uniform distribution on the interval [0, 1] is a special case of the Beta distribution with $\alpha = \beta = 1$.
- [Dirichlet Distribution](dirichlet_distribution.md) - The Dirichlet distribution is the generalization of the Beta to higher dimensions.
```

## References

- Wikipedia - [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)