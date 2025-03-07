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
# Half-Cauchy Distribution

<audio controls> <source src="../../_static/halfcauchy.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Non-Negative](../../gallery_tags.rst#non-negative), [Asymmetric](../../gallery_tags.rst#asymmetric), [Heavy-tailed](../../gallery_tags.rst#heavy-tailed)

The Half-Cauchy distribution is a continuous probability distribution that is derived from the Cauchy distribution but is restricted to only positive values. It is characterized by a single scale parameter ($\beta$), which determines the width of the distribution. Similar to the Cauchy distribution, the Half-Cauchy distribution has undefined mean and variance, making it an example of a "pathological" distribution with heavy tails.

In Bayesian statistics, the Half-Cauchy distribution is often used as a prior for scale parameters.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in [0, \infty)`
Mean      undefined
Variance  undefined
========  ==========================================
```

**Parameters:**

- $\beta$ : (float) Scale parameter, $\beta > 0$.

### Probability Density Function (PDF)

$$ 
f(x|\beta) = \frac{2}{\pi \beta \left[1 + \left(\frac{x}{\beta}\right)^2\right]}
$$

```{code-cell}
---
tags: [remove-input]
mystnb: image
---
from preliz import HalfCauchy, style
style.use('preliz-doc')
betas = [0.5, 1, 2]
for beta in betas:
    HalfCauchy(beta).plot_pdf(support=(0, 5))
```

## Cumulative Distribution Function (CDF):

$$ 
F(x|\beta) = \frac{2}{\pi} \arctan\left(\frac{x}{\beta}\right)
$$

```{code-cell} 
---
tags: [remove-input]
mystnb: image
---
for beta in betas:
    HalfCauchy(beta).plot_cdf(support=(0, 5))
```

```{seealso} 
:class: seealso

**Common Alternatives:**

- [Half-Student's t](halfstudentt.md) - The Half-Cauchy distribution is a special case of the Half-Student's t-distribution with $\nu=1$.
- [Half-Normal](halfnormal.md) - A distribution that considers only the positive half of the normal distribution.

**Related Distributions:**

- [Cauchy](cauchy.md) - The Cauchy distribution is the parent distribution from which the Half-Cauchy is derived.
 ```
## References

- [Wikipedia - Cauchy](https://en.wikipedia.org/wiki/Cauchy_distribution)