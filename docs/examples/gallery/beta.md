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
# Beta Distribution

<audio controls> <source src="../../_static/beta.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Beta distribution is a continuous probability distribution bounded between 0 and 1. It is usually defined by two positive shape parameters: ($\alpha$) and ($\beta$). But other parametrization like mean ($\mu$) and concentration ($\nu$) are also common.

The Beta distribution can adopt a wide range of "shapes" including uniform, U-shape, normal-like, exponential-like, and many others, always restricted to the unit interval. This flexibility makes it a versatile choice for modeling random variables that represent proportions, probabilities, or rates. For example, the Beta distribution is commonly used to model the uncertainty of the true proportion of successes in a series of Bernoulli trials, where $\alpha$ and $\beta$ represent the number of successes and failures, respectively.

## Key properties and parameters

```{eval-rst}
========  ==============================================================
Support   :math:`x \in (0, 1)`
Mean      :math:`\dfrac{\alpha}{\alpha + \beta}`
Variance  :math:`\dfrac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`
========  ==============================================================
```

**Parameters:**

- $\alpha$ : (float) Shape parameter, $\alpha > 0$.
- $\beta$ : (float) Shape parameter, $\beta > 0$.
- $\mu$ : (float) Mean of the distribution, $0 < \mu < 1$.
- $\sigma$ : (float) Standard deviation of the distribution, $\sigma < sqrt(\mu(1-\mu))$.
- $\nu$ : (float) Concentration parameter, $\nu > 0$.

**Alternative parameterization**

The Beta distribution has 3 alternative parameterizations. In terms of $\alpha$ and $\beta$, $\mu$ and $\sigma$, and $\mu$ and $\nu$. 

The link between the parameters is given by:

$$
\alpha = \mu \nu \\
\beta = (1 - \mu) \nu
$$

### Probability Density Function (PDF)

$$
f(x \mid \alpha, \beta) =
    \frac{x^{\alpha - 1} (1 - x)^{\beta - 1}}{B(\alpha, \beta)}
$$

where $B(\alpha,\beta)$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function) 

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\alpha$ and $\beta$
:sync: alpha-beta
```{jupyter-execute}
:hide-code:

from preliz import Beta, style
style.use('preliz-doc')
alphas = [.5, 5., 2.]
betas = [.5, 5., 5.]
for alpha, beta in zip(alphas, betas):
    ax = Beta(alpha, beta).plot_pdf()
ax.set_ylim(0, 5);
```
:::::

:::::{tab-item} Parameters $\mu$ and $\sigma$  
:sync: mu-sigma

```{jupyter-execute}
:hide-code:

mus = [0.5, 0.5, 0.286]
sigmas = [0.3536, 0.1507, 0.1598]
for mu, sigma in zip(mus, sigmas):
    ax = Beta(mu=mu, sigma=sigma).plot_pdf()
ax.set_ylim(0, 5);
```
:::::

:::::{tab-item} Parameters $\mu$ and $\nu$
:sync: mu-nu

```{jupyter-execute}
:hide-code:

nus = [1.0, 10.0, 7.0]
for mu, nu in zip(mus, nus):
    ax = Beta(mu=mu, nu=nu).plot_pdf()
ax.set_ylim(0, 5);
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x \mid \alpha,\beta) = \frac{B(x;\alpha,\beta)}{B(\alpha,\beta)} = I_x(\alpha,\beta)
$$

where $B(x;\alpha,\beta)$ is the [Incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function) and $I_x(\alpha,\beta)$ is the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\alpha$ and $\beta$
:sync: alpha-beta

```{jupyter-execute}
:hide-code:
for alpha, beta in zip(alphas, betas):
    ax = Beta(alpha, beta).plot_cdf()
```
:::::

:::::{tab-item} Parameters $\mu$ and $\sigma$  
:sync: mu-sigma

```{jupyter-execute}
:hide-code:
for mu, sigma in zip(mus, sigmas):
    ax = Beta(mu=mu, sigma=sigma).plot_cdf()
```
:::::

:::::{tab-item} Parameters $\mu$ and $\nu$
:sync: mu-nu

```{jupyter-execute}
:hide-code:
for mu, nu in zip(mus, nus):
    ax = Beta(mu=mu, nu=nu).plot_cdf()
```
:::::
::::::

```{seealso}
:class: seealso

**Common Alternatives:**

- [Kumaraswamy](kumaraswamy.md) -  It is similar to the Beta distribution, but with closed form expression for its probability density function, cumulative distribution function and quantile function.

**Related Distributions:**
- [Beta Scaled](beta_scaled.md) - A Beta distribution defined on an arbitrary range.
- [Uniform](uniform.md) - The Uniform distribution on the interval [0, 1] is a special case of the Beta distribution with $\alpha = \beta = 1$.
- [Dirichlet](dirichlet.md) - The Dirichlet distribution is the generalization of the Beta to higher dimensions.
```

## References

- Wikipedia - [Beta distribution](https://en.wikipedia.org/wiki/Beta_distribution)