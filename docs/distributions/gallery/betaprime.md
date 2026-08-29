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
# Beta Prime Distribution

<audio controls> <source src="../../_static/betaprime.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Non-Negative](../../gallery_tags.rst#non-negative), [Asymmetric](../../gallery_tags.rst#asymmetric)

The Beta prime distribution is a continuous probability distribution defined on the positive real line. It is usually defined by two positive shape parameters: ($\alpha$) and ($\beta$). An alternative parametrization in terms of mean ($\mu$) and precision ($\nu$) is also common.

The Beta prime distribution is closely related to the Beta distribution: if $X$ follows a Beta$(\alpha, \beta)$ distribution, then $Y = X / (1 - X)$ follows a Beta prime$(\alpha, \beta)$ distribution. 
Because of this relationship, the Beta prime distribution is often used to model ratios or odds, such as the odds of success in a series of Bernoulli trials, 
where the Beta distribution would be used to model the probability of success itself. It also shows in the definition of priors like [R²D²](https://n-kall.github.io/priorDB/gen_linear_regression/r2d2.html), as $\tau^2=\frac{R^2}{1-R^2}$.

## Key properties and parameters

```{eval-rst}
========  ==========================================================================
Support   :math:`x \in (0, \infty)`
Mean      :math:`\dfrac{\alpha}{\beta - 1}` for :math:`\beta > 1`
Variance  :math:`\dfrac{\alpha(\alpha + \beta - 1)}{(\beta - 1)^2(\beta - 2)}` for :math:`\beta > 2`
========  ==========================================================================
```

**Parameters:**

- $\alpha$ : (float) Shape parameter, $\alpha > 0$.
- $\beta$ : (float) Shape parameter, $\beta > 0$.
- $\mu$ : (float) Mean of the distribution, $\mu > 0$.
- $\nu$ : (float) Precision parameter, $\nu > 0$.

**Alternative parameterization**

The Beta prime distribution has 2 alternative parameterizations. In terms of $\alpha$ and $\beta$, or $\mu$ and $\nu$.

The link between the parameters is given by:

$$
\begin{align}
\alpha &= \mu (1 + \nu) \\
\beta &= 2 + \nu
\end{align}
$$

### Probability Density Function (PDF)

$$
f(x \mid \alpha, \beta) =
    \frac{x^{\alpha - 1} (1 + x)^{-\alpha - \beta}}{B(\alpha, \beta)}
$$

where $B(\alpha,\beta)$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function)

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\alpha$ and $\beta$
:sync: alpha-beta
```{jupyter-execute}
:hide-code:

from preliz import BetaPrime, style
style.use('preliz-doc')
alphas = [.5, 5., 2.]
betas = [.5, 5., 5.]
for alpha, beta in zip(alphas, betas):
    ax = BetaPrime(alpha, beta).plot_pdf(support=(0, 5))
```
:::::

:::::{tab-item} Parameters $\mu$ and $\nu$
:sync: mu-nu

```{jupyter-execute}
:hide-code:

mus = [1., 1., 0.6]
nus = [1., 8., 6.]
for mu, nu in zip(mus, nus):
    ax = BetaPrime(mu=mu, nu=nu).plot_pdf(support=(0, 5))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x \mid \alpha,\beta) = I_{\frac{x}{1+x}}(\alpha,\beta)
$$

where $I_z(\alpha,\beta)$ is the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\alpha$ and $\beta$
:sync: alpha-beta

```{jupyter-execute}
:hide-code:
for alpha, beta in zip(alphas, betas):
    ax = BetaPrime(alpha, beta).plot_cdf(support=(0, 5))
```
:::::

:::::{tab-item} Parameters $\mu$ and $\nu$
:sync: mu-nu

```{jupyter-execute}
:hide-code:
for mu, nu in zip(mus, nus):
    ax = BetaPrime(mu=mu, nu=nu).plot_cdf(support=(0, 5))
```
:::::
::::::

```{seealso}
:class: seealso

**Related Distributions:**
- [Beta](beta.md) - If $X$ follows a Beta$(\alpha, \beta)$ distribution, then $X / (1 - X)$ follows a Beta prime$(\alpha, \beta)$ distribution.
- [Gamma](gamma.md) - The Beta prime distribution arises as the ratio of two independent Gamma random variables with shape parameters $\alpha$ and $\beta$ (and equal scale).
```

## References

- Wikipedia - [Beta prime distribution](https://en.wikipedia.org/wiki/Beta_prime_distribution)