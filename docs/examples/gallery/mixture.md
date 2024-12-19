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
(Mixture_gallery)=
# Mixture Distribution

<audio controls> <source src="../../_static/mixture.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

This is not a distribution per se, but a modifier of univariate distributions.

A mixture distribution is a probability distribution that results from the combination of two or more univariate distributions. The resulting distribution is a weighted sum of the component distributions. 

Mixture distributions are widely used whenever a single, simple distribution cannot adequately capture the complexity of a dataset. For example, in modeling the distribution of incomes across a population with distinct socioeconomic groups, mixtures can represent each subgroup’s unique income pattern. 

## Key properties and parameters

**Parameters:**

- `dists` : (list of Univariate PreLiz distributions) Components of the mixture. They should be all discrete or all continuous.
- `weights` : (list of floats) List of weights for each distribution. Weights must be larger or equal to 0 and their sum must be positive. If the weights do not sum up to 1, they will be normalized.

### Probability Density Function (PDF)

Given a list of base distributions with cumulative distribution functions (CDFs) and probability density/mass functions (PDFs/PMFs). The pdf of a Mixture distribution is:

$$
f(x) = \sum_{i=1}^n \, w_i \, p_i(x)
$$

where $w_i$ is the weight of the $i$-th distribution and $p_i(x)$ is the PDF/PMF of the $i$-th distribution.

```{code-cell}
---
tags: [remove-input]
mystnb:
    image:
        alt: Mixture Distribution PDF
---

from preliz import Normal, Mixture, style
style.use('preliz-doc')

Mixture([Normal(0, 0.5), Normal(2, 0.5)], [0.2, 0.8]).plot_pdf()
Normal(0, 0.5).plot_pdf(alpha=0.5)
Normal(2, 0.5).plot_pdf(alpha=0.5);
```

### Cumulative Distribution Function (CDF)

The cumulative distribution function (CDF) of a Mixture distribution is the sum of the CDFs of the component distributions.

$$
F(x) = \sum_{i=1}^n \, w_i \, P_i(x)
$$

where $w_i$ is the weight of the $i$-th distribution and $P_i(x)$ is the CDF of the $i$-th distribution.

```{code-cell}
---
tags: [remove-input]
mystnb:
    image:
        alt: Mixture Distribution CDF
---

Mixture([Normal(0, 0.5), Normal(2, 0.5)], [0.2, 0.8]).plot_cdf()
Normal(0, 0.5).plot_cdf(alpha=0.5)
Normal(2, 0.5).plot_cdf(alpha=0.5);
```

```{seealso}
:class: seealso

**Related Distributions:**

- [Hurdle](hurdle.md) -  A modifier that combines a point mass at zero with a separate distribution for positive outcomes.
```

## References

- Wikipedia - [Mixture distribution](https://en.wikipedia.org/wiki/Mixture_distribution)