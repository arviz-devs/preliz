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
# Triangular Distribution

<audio controls> <source src="../../_static/triangular.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Bounded](../../gallery_tags.rst#bounded), [Light-tailed](../../gallery_tags.rst#light-tailed)

The Triangular distribution is a continuous probability distribution with a triangular shape. It is defined by three parameters: the lower bound $lower$, the mode $c$, and the upper bound $upper$. Where $lower \leq c \leq upper$. 

The Triangular distribution is often called a "lack of knowledge" distribution because it can be used when there is no prior knowledge about the distribution of the random variable other than the minimum, maximum, and most likely values. It is often used in business simulations and decision-making, project management, and audio dithering.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in [lower, upper]`
Mean      :math:`\dfrac{lower + upper + c}{3}`
Variance  :math:`\dfrac{upper^2 + lower^2 +c^2 - lower*upper - lower*c - upper*c}{18}`
========  ==========================================
```

**Parameters:**

- $lower$ : (float) Lower bound of the distribution.
- $c$ : (float) Mode of the distribution, $lower \leq c \leq upper$.
- $upper$ : (float) Upper bound of the distribution.

### Probability Density Function (PDF)

$$
f(x|lower, c, upper) =
    \begin{cases}
        0 & \text{for } x < lower \\
        \frac{2(x - lower)}{(upper - lower)(c - lower)} & \text{for } lower \leq x < c \\
        \frac{2}{upper - lower} & \text{for } x = c \\
        \frac{2(upper - x)}{(upper - lower)(upper - c)} & \text{for } c < x \leq upper \\
    \end{cases}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Triangular Distribution PDF
---

from preliz import Triangular, style
style.use('preliz-doc')
lowers = [0., -1, 2]
cs = [2., 0., 6.5]
uppers = [4., 1, 8]
for lower, c, upper in zip(lowers, cs, uppers):
    scale = upper - lower
    c_ = (c - lower) / scale
    Triangular(lower, c, upper).plot_pdf()
```

### Cumulative Distribution Function (CDF)

$$
F(x|lower, c, upper) =
    \begin{cases}
        0 & \text{for } x < lower \\
        \frac{(x - lower)^2}{(upper - lower)(c - lower)} & \text{for } lower \leq x < c \\
        1 - \frac{(upper - x)^2}{(upper - lower)(upper - c)} & \text{for } c \leq x \leq upper \\
        1 & \text{for } x \geq upper
    \end{cases}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Triangular Distribution CDF
---

for lower, c, upper in zip(lowers, cs, uppers):
    scale = upper - lower
    c_ = (c - lower) / scale
    Triangular(lower, c, upper).plot_cdf()
```

```{seealso}
:class: seealso

**Related Distributions:**

- [Uniform Distribution](uniform.md) - A distribution with constant probability that is also used when there is limited prior knowledge about the distribution of the random variable: in this case, only the $lower$ and $upper$ bounds are known.
```

## References

- Wikipedia. [Triangular distribution](https://en.wikipedia.org/wiki/Triangular_distribution).