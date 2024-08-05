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

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
import matplotlib.pyplot as plt

from preliz import Cauchy, style
style.use('preliz-doc')
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

- [Student's t](students_t.md) - The Cauchy distribution is a special case of the Student's t-distribution with $\nu=1$.
- [Normal](normal.md) - Often used as an alternative to the Cauchy distribution when the tails are not of primary concern.
- [Logistic](logistic.md) - A symmetric distribution with lighter tails than the Cauchy distribution.

**Related Distributions:**

- [Half-Cauchy](half_cauchy.md) - Considers only the positive half of the Cauchy distribution, often used as a prior for scale parameters.
```

## References

- [Wikipedia - Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)




