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
# Inverse Gamma Distribution

<audio controls> <source src="../../_static/inversegamma.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Asymmetric](../../gallery_tags.rst#asymmetric), [Non-Negative](../../gallery_tags.rst#non-negative), [Heavy-tailed](../../gallery_tags.rst#heavy-tailed)

The Inverse Gamma distribution is a continuous probability distribution defined as the distribution of the reciprocal of a Gamma-distributed random variable. It is characterized by two parameters: the shape parameter $\alpha$ and the scale parameter $\beta$.

In mathematics, the Lévy distribution (a special case of the inverse gamma distribution with a shape parameter $\alpha=0.5$) describes the hitting time of a Wiener process, which is the probability distribution of the first time this stochastic process reaches a specific level.

In Bayesian statistics, the Inverse Gamma distribution often appears as the marginal posterior distribution for an unknown variance in a normal distribution when using an uninformative prior. It also serves as an analytically tractable conjugate prior when an informative prior is required.

## Key properties and parameters

```{eval-rst}
========  ===============================
Support   :math:`x \in (0, \infty)`
Mean      :math:`\dfrac{\beta}{\alpha-1}` for :math:`\alpha > 1`
Variance  :math:`\dfrac{\beta^2}{(\alpha-1)^2(\alpha - 2)}` for :math:`\alpha > 2`
========  ===============================
```

**Parameters:**

- $\alpha$ : (float) Shape parameter, $\alpha > 0$.
- $\beta$ : (float) Scale parameter, $\beta > 0$.

**Alternative parametrization**

The Inverse Gamma distribution has 2 alternative parametrizations: in terms of the shape parameter $\alpha$ and the scale parameter $\beta$, or in terms of $\mu$ (mean) and $\sigma$ (standard deviation). 

The link between the parameters is given by:

$$
\alpha = \frac{\mu^2}{\sigma^2} + 2 \\
\beta = \frac{\mu^3}{\sigma^2} + \mu
$$

### Probability Density Function (PDF)

$$
f(x; \alpha, \beta) = \dfrac{\beta^\alpha}{\Gamma(\alpha)} x^{-\alpha-1}\exp(-\dfrac{\beta}{x})
$$

where $\Gamma(\alpha)$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\alpha$ and $\beta$
:sync: alpha-beta
```{jupyter-execute}
:hide-code:

from preliz import InverseGamma, style
style.use('preliz-doc')
alphas= [3, 4, 5]
betas = [1, 1, 0.5]

for alpha, beta in zip(alphas, betas):
    ax = InverseGamma(alpha, beta).plot_pdf(support=(0, 3))
```
:::::

:::::{tab-item} Parameters $\mu$ and $\sigma$
:sync: mu-sigma

```{jupyter-execute}
:hide-code:

mus = [0.5, 0.33, 0.125]
sigmas = [0.5, 0.236, 0.072]

for mu, sigma in zip(mus, sigmas):
    ax = InverseGamma(mu=mu, sigma=sigma).plot_pdf(support=(0, 3))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x; \alpha, \beta) = \dfrac{\Gamma(\alpha, \dfrac{\beta}{x})}{\Gamma(\alpha)}
$$

where $\Gamma(\alpha, x)$ is the [upper incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\alpha$ and $\beta$
:sync: alpha-beta
```{jupyter-execute}
:hide-code:

for alpha, beta in zip(alphas, betas):
    ax = InverseGamma(alpha, beta).plot_cdf()
```
:::::

:::::{tab-item} Parameters $\mu$ and $\sigma$
:sync: mu-sigma

```{jupyter-execute}
:hide-code:

for mu, sigma in zip(mus, sigmas):
    ax = InverseGamma(mu=mu, sigma=sigma).plot_cdf()
```
:::::
::::::

```{seealso}
:class: seealso

**Related Distributions:**
- [Gamma Distribution](gamma.md) - The Gamma distribution is the reciprocal of the Inverse Gamma distribution.
```

## References

- [Wikipedia - Inverse Gamma Distribution](https://en.wikipedia.org/wiki/Inverse-gamma_distribution)