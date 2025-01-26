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
# Kumaraswamy Distribution

<audio controls> <source src="../../_static/kumaraswamy.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Kumaraswamy distribution is a continuous probability distribution bounded between 0 and 1. It is characterized by two positive shape parameters: $a$ and $b$. 

The Kumaraswamy distribution is a flexible distribution that can adopt a wide range of shapes, including uniform, U-shape, exponential-like, and many others, always restricted to the unit interval [0, 1].

## Key properties and parameters:

```{eval-rst}
========  ==============================================================
Support   :math:`x \in (0, 1)`
Mean      :math:`b B(1 + \tfrac{1}{a}, b)`
Variance  :math:`b B(1 + \tfrac{2}{a}, b) - (b B(1 + \tfrac{1}{a}, b))^2`
========  ==============================================================
```

**Parameters:**

- $a$ : (float) First shape parameter, $a > 0$.
- $b$ : (float) Second shape parameter, $b > 0$.

### Probability Density Function (PDF)

$$
f(x|a, b) = a b x^{a-1} (1 - x^{a})^{b-1}
$$

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
import matplotlib.pyplot as plt

from preliz import Kumaraswamy, style
style.use('preliz-doc')

a_s = [.5, 5., 1., 1., 2., 2.]
b_s = [.5, 1., 1., 3., 2., 5.]

for a, b in zip(a_s, b_s):
    ax = Kumaraswamy(a, b).plot_pdf()
    ax.set_ylim(0, 3.)
```

### Cumulative Distribution Function (CDF)

$$
F(x|a, b) = 1 - (1 - x^{a})^{b}
$$

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
for a, b in zip(a_s, b_s):
    ax = Kumaraswamy(a, b).plot_cdf()
```

```{seealso}
:class: seealso

**Common Alternatives:**

- [Beta](beta.md) - The Kumaraswamy distribution is similar to the Beta distribution, but with closed-form expressions  for its probability density function, cumulative distribution function and quantile function.

**Related Distributions:**

- [Uniform](uniform.md) - The Uniform distribution on the interval [0, 1] is a special case of the Kumaraswamy distribution with $a = b = 1$.
```

## References

- [Wikipedia - Kumaraswamy distribution](https://en.wikipedia.org/wiki/Kumaraswamy_distribution)