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
(Hurdle_gallery)=
# Hurdle Distribution

<audio controls> <source src="../../_static/hurdle.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

This is not a distribution per se, but a modifier of univariate distributions.


The hurdle distribution is a mixture distribution that combines a point mass at zero with a continuous distribution. It is used to model data that is a mixture of two processes: one that generates zeros and another that generates non-zero values. The hurdle distribution is commonly used in econometrics to model data with excess zeros, such as healthcare costs, insurance claims, and counts of events.

## Key properties and parameters

**Parameters:**

- `dist` : (PreliZ distribution) Univariate PreliZ distribution which will be truncated at zero.
- `psi` : (float) Expected proportion of the base distribution (0 < psi < 1)

### Probability Density Function (PDF)

Given a base distribution with parameters $\theta$, cumulative distribution function (CDF) and probability density/mass function (PDF). The density of a Hurdle distribution is:

$$
f(x \mid \psi, \mu) =
    \left\{
        \begin{array}{l}
            (1 - \psi), \text{if } x = 0 \\
            \psi
            \frac{\text{PDF}(x \mid \theta)}
            {1 - \text{CDF}(\epsilon \mid \theta)}, \text{if } x \neq= 0,\ldots
        \end{array}
    \right.
$$

where $\psi$ is the expected proportion of the base distribution and $\epsilon$ is the machine precision for continuous distribution or 0 for discrete ones.

The following figure shows the difference between a Gamma distribution and a HurdleGamma, with the same parameters for the base distribution (Gamma).

```{code-cell}
---
tags: [remove-input]
mystnb:
    image:
        alt: Hurdle Distribution PDF
---

from preliz import Gamma, Hurdle, style
style.use('preliz-doc')
Hurdle(Gamma(mu=2, sigma=1), 0.8).plot_pdf()
Gamma(mu=2, sigma=1).plot_pdf();

```

### Cumulative Distribution Function (CDF)

$$
F(x \mid \psi, \mu) =
\left\{
    \begin{array}{ll}
        0, \text{if } x < 0 \\
        1 - \psi, \text{if } x = 0 \\
        1 - \psi + \psi \cdot \frac{\text{PoissonCDF}(x \mid \mu) - \text{PoissonCDF}(0 \mid \mu)}{1 - \text{PoissonCDF}(0 \mid \mu)}, \text{if } x = 1, 2, 3, \ldots
    \end{array}
\right.
$$



```{code-cell}
---
tags: [remove-input]
mystnb:
    image:
        alt: Hurdle Distribution CDF
---

Hurdle(Gamma(mu=2, sigma=1), 0.8).plot_cdf()
Gamma(mu=2, sigma=1).plot_cdf();
```

```{seealso}
:class: seealso

**Related Distributions:**

- [Mixture](mixture.md) - A distribution modifier that combines two or more distributions with weights.
- [ZeroInflatedPoisson](zero_inflated_poisson.md) - A distribution that extends the Poisson distribution by adding a mechanism to account for excess zeros, often observed in count data.
- [Zero-Inflated Binomial Distribution](zeroinflatedbinomial.md) - A distribution that combines the Binomial distribution with a zero-inflation component, useful for modeling count data with an excess of zeros.
- [Zero-Inflated Negative Binomial Distribution](zeroinflatednegativebinomial.md)- A distribution that combines the Negative Binomial distribution with a zero-inflation component, modeling scenarios with overdispersion and an excess of zeros.
```

## References

- [Wikipedia - Hurdle Distribution](https://en.wikipedia.org/wiki/Hurdle_model)