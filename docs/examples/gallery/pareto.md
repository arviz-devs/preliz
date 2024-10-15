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
# Pareto Distribution

<audio controls> <source src="../../_static/pareto.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Pareto distribution is a continuous probability distribution that was introduced by the Italian scientist Vilfredo Pareto (1848–1923). It is a power-law probability distribution that is characterized by a heavy right tail. It is defined by two parameters: the scale parameter $m$ and the shape parameter $\alpha$.

It was originally used to describe the distribution of wealth in society where a small proportion of the population holds a large proportion of the wealth (the "80-20 rule"). It has since been used in various fields to describe a wide range of phenomena where events get rarer at greater magnitudes.

In Bayesian modeling, it plays a key role in Pareto Smoothed Importance Sampling (PSIS) for stabilizing importance weights. PSIS is used in model evaluation methods like PSIS-LOO (Leave-One-Out Cross-Validation) where LOO applies a smoothing procedure that uses values from an estimated Pareto distribution to replace the largest importance weights, making LOO more robust. Moreover, the estimated $\hat{k}$ parameter of the Pareto distribution can be used as a  diagnostic tool to assess the reliability of the importance weights.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Pareto Distribution PDF
---

from preliz import Pareto, style
style.use('preliz-doc')
alphas = [1., 5., 5.]
ms = [1., 1., 2.]
for alpha, m in zip(alphas, ms):
    Pareto(alpha, m).plot_pdf(support=(0,4))
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Pareto Distribution CDF
---

for alpha, m in zip(alphas, ms):
    Pareto(alpha, m).plot_cdf(support=(0,4))
```

## Key properties and parameters:

```{eval-rst}

========  ==========================================
Support   :math:`x \in [m, \infty)`
Mean      :math:`\frac{\alpha m}{\alpha - 1}` for :math:`\alpha > 1`
Variance  :math:`\frac{m^2 \alpha}{(\alpha - 1)^2 (\alpha - 2)}` for :math:`\alpha > 2`
========  ==========================================
```

**Probability Density Function (PDF):**

$$
f(x|\alpha, m) = \frac{\alpha m^\alpha}{x^{\alpha + 1}}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x|\alpha, m) = 1 - \left(\frac{m}{x}\right)^\alpha
$$

```{seealso}
:class: seealso

**Related Distributions:**

- [Exponential Distribution](exponential.md) - If X is Pareto distributed, with scale parameter $m$, and shape parameter $\alpha$, then $Y = log(X/m)$ is exponentially distributed with rate parameter $\lambda = \alpha$.
- [Log-Normal Distribution](log_normal.md) - Also used for modeling positive, skewed data with long tails. The log-normal distribution allows for more flexibility in shaping the tail behavior compared to the Pareto distribution.
```

## References

- [Wikipedia - Pareto Distribution](https://en.wikipedia.org/wiki/Pareto_distribution)

