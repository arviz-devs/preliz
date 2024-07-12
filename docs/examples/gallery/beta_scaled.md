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
# Beta Scaled Distribution

The Beta scaled distribution is a continuous probability distribution similar to the Beta distribution but instead of being bounded between 0 and1 it is bounded between $lower$ and $upper$. It is usually defined by two positive shape parameters: ($\alpha$) and ($\beta$). But other parametrization like mean ($\mu$) and concentration ($\nu$) are also common.

The Beta scaled distribution can adopt a wide range of "shapes" including uniform, U-shape, normal-like, exponential-like, and many others, always restricted to a given interval. This flexibility makes it a versatile choice for modeling random variables that are known to be bounded like percentages, grades, some physical quantities like temperature of liquid water at a given pressure. 

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Beta Scaled Distribution PDF
---

import arviz as az
from preliz import BetaScaled
az.style.use('arviz-doc')
alphas = [2, 2]
betas = [2, 5]
lowers = [-0.5, -1]
uppers = [1.5, 2]
for alpha, beta, lower, upper in zip(alphas, betas, lowers, uppers):
    BetaScaled(alpha, beta, lower, upper).plot_pdf()
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Beta Scaled Distribution CDF
---

for alpha, beta, lower, upper in zip(alphas, betas, lowers, uppers):
    BetaScaled(alpha, beta, lower, upper).plot_cdf()
```


## Key properties and parameters:

```{eval-rst}
========  ============================================================================
Support   :math:`x \in (lower, upper)`
Mean      :math:`\dfrac{\alpha}{\alpha + \beta} (upper-lower) + lower`
Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)} (upper-lower)`
========  ============================================================================
```

**Probability Density Function (PDF):**


$$
f(x \mid \alpha, \beta, lower, upper) =
    \frac{(x-\text{lower})^{\alpha - 1} (\text{upper} - x)^{\beta - 1}}
    {(\text{upper}-\text{lower})^{\alpha+\beta-1} B(\alpha, \beta)}
$$

where $B(\alpha,\beta)$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function) 

**Cumulative Distribution Function (CDF):**

$$
F(x \mid \alpha,\beta, lower, upper) = \frac{B(y;\alpha,\beta)}{B(\alpha,\beta)} = I_y(\alpha,\beta)
$$

where $y$ is the scaled variable $y = \frac{(x - lower)}{(upper - lower)}$. $B(x;\alpha,\beta)$ is the [Incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function) and $I_x(\alpha,\beta)$ is the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function).



```{seealso}
:class: seealso

**Related Distributions:**
- [Beta](beta_scaled_distribution.md) - A Beta scaled distribution with $lower=0$ and $upper=1$.
- [Kumaraswamy Distribution](Kumaraswamy_distribution.md) -  It is similar to the Beta scaled distribution, but restricted to the [0, 1] interval and with closed form expression for its probability density function, cumulative distribution function and quantile function.
- [Uniform Distribution](uniform_distribution.md) - The Uniform distribution on the interval $[lower, upper]$ is a special case of the Beta scaled distribution with $\alpha = \beta = 1$.
```

## References

- Wikipedia - [Beta distribution with four parameters](https://en.wikipedia.org/wiki/Beta_distribution#Four_parameters)