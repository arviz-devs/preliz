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
# Dirichlet Distribution

<audio controls> <source src="../../_static/dirichlet.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Dirichlet distribution is a multivariate continuous probability distribution defined on the standard $K$-simplex, where $K \geq 2$. It is parameterized by a vector of positive concentration parameters $\alpha = (\alpha_1, \alpha_2, \ldots, \alpha_K)$, which control the shape of the distribution. The support of the Dirichlet distribution consists of vectors whose entries sum to one, making it particularly useful for modeling probabilities or proportions.

A key property of the Dirichlet distribution is that it generalizes the Beta distribution to multiple dimensions. When used as a prior distribution in Bayesian statistics, it becomes the conjugate prior of both the categorical and multinomial distributions. This conjugacy simplifies inference, particularly in hierarchical models and Bayesian mixture models. For example, the posterior distribution of a categorical model with a Dirichlet prior remains Dirichlet after observing new data.

## Key properties and parameters

```{eval-rst}
========  ===============================================
Support   :math:`x_i \in (0, 1)` for :math:`i \in \{1, \ldots, K\}` such that :math:`\sum x_i = 1`
Mean      :math:`\dfrac{a_i}{\sum a_i}`
Variance  :math:`\dfrac{a_i - \sum a_0}{a_0^2 (a_0 + 1)}` where :math:`a_0 = \sum a_i`
========  ===============================================
```

**Parameters:**

- $\alpha$ : (array of floats) Concentration parameters $\alpha = (\alpha_1, \alpha_2, \ldots, \alpha_K)$ where $\alpha_i > 0$.

### Probability Density Function (PDF)

$$
f(\mathbf{x}|\mathbf{a}) =
    \frac{\Gamma(\sum_{i=1}^k a_i)}{\prod_{i=1}^k \Gamma(a_i)}
    \prod_{i=1}^k x_i^{a_i - 1}
$$

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Dirichlet Distribution PDF
---
import logging
import matplotlib.pyplot as plt
from preliz import Dirichlet


logger = logging.getLogger()
logger.disabled = True

_, axes = plt.subplots(2, 2)
alphas = [[0.5, 0.5, 0.5], [1, 1, 1], [5, 5, 5], [5, 2, 1]]
for alpha, ax in zip(alphas, axes.ravel()):
    Dirichlet(alpha).plot_pdf(marginals=False, ax=ax)

logger.disabled = False
```

### Cumulative Distribution Function (CDF)


The Dirichlet joint CDF does not have a known analytical solution due to the complexity of integrating its multivariate density over the simplex. However, each marginal distribution is a Beta distribution with a well-defined CDF that can be computed directly.

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Dirichlet Distribution CDF
---

for alpha in alphas:
    Dirichlet(alpha).plot_cdf()
```

```{seealso}
:class: seealso

**Common Alternatives:**

- [Beta Distribution](beta.md) - When $K = 2$, the Dirichlet distribution reduces to the Beta distribution with $\alpha = \alpha_1$ and $\beta = \alpha_2$.

**Related Distributions:**

- [Gamma Distribution](gamma.md) -  A Dirichlet distribution can be constructed from a normalized set of $K$ independent gamma-distributed random variables.
- [Categorical Distribution](categorical.md) - A discrete distribution for modeling outcomes with multiple categories, often used in conjunction with a Dirichlet prior in Bayesian settings.
```

## References

- [Wikipedia - Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
