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
# Cauchy Distribution

The Cauchy distribution is a continuous probability distribution with a bell-shaped curve and heavy tails. It is defined by two parameters: the location parameter ($\alpha$), which specifies the peak of the distribution, and the scale parameter ($\beta$) which determines the width. The Cauchy distribution has no defined mean or variance because the tails of the distribution decay so slowly that the integrals defining the mean and variance do not converge, making it an example of a distribution with undefined moments (often termed a "pathological" distribution).

In physics, the Cauchy distribution is also known as the Lorentz distribution, Cauchy–Lorentz distribution, Lorentzian function, or Breit–Wigner distribution. It is used to describe the shape of spectral lines in spectroscopy. Specifically, the intercept on the x-axis of a beam of light coming from a point with location and scale parameters ($\alpha$, $\beta$) is modeled as Cauchy distributed. This distribution is particularly suited for such applications due to its heavy tails, which accurately represent the likelihood of observing extreme deviations in the spectral line shape.

In finance, the Cauchy distribution is used to model extreme events, such as stock market crashes, where the tails of the distribution are of particular interest.

In Bayesian modeling, the Cauchy distribution has been [recommended](https://doi.org/10.1214/08-AOAS191) as a prior for logistic regression coefficients.

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
import matplotlib.pyplot as plt
import arviz as az
from preliz import Cauchy
az.style.use('arviz-doc')
alphas = [0., 0., -2.]
betas = [1, 0.5, 1]
for alpha, beta in zip(alphas, betas):
    Cauchy(alpha, beta).plot_pdf(support=(-5, 5))
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
for alpha, beta in zip(alphas, betas):
    Cauchy(alpha, beta).plot_cdf(support=(-5, 5))
```

## Key properties and parameters:

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      undefined
Variance  undefined
========  ==========================================
```

**Probability Density Function (PDF):**

$$
f(x|\alpha, \beta) = \frac{1}{\pi \beta \left[1 + \left(\frac{x - \alpha}{\beta}\right)^2\right]}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x|\alpha, \beta) = \frac{1}{2} + \frac{1}{\pi} \arctan\left(\frac{x - \alpha}{\beta}\right)
$$

```{seealso}
:class: seealso

**Common Alternatives:**

- [Student's t Distribution](students_t_distribution.md) - The Cauchy distribution is a special case of the Student's t-distribution with $\nu=1$.
- [Normal Distribution](normal_distribution.md) - Often used as an alternative to the Cauchy distribution when the tails are not of primary concern.
- [Logistic Distribution](logistic_distribution.md) - A symmetric distribution with lighter tails than the Cauchy distribution.

**Related Distributions:**

- [Half-Cauchy Distribution](half_cauchy_distribution.md) - Considers only the positive half of the Cauchy distribution, often used as a prior for scale parameters.
```

## References

- [Wikipedia - Cauchy Distribution](https://en.wikipedia.org/wiki/Cauchy_distribution)




