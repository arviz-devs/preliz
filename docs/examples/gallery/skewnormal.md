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
# Skew-Normal Distribution

<audio controls> <source src="../../_static/skewnormal.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The skew-normal distribution is a continuous probability distribution that generalizes the normal distribution by introducing a skewness parameter ($\alpha$). This parameter allows the distribution to be asymmetric around its mean, with $\alpha$ determining the direction and degree of the asymmetry. When $\alpha = 0$, the skew-normal distribution reduces to the normal distribution.

The skew-normal distribution is often used to model data that exhibit skewness, such as financial returns, income distributions, and reaction times. In these cases, the skew-normal distribution provides a flexible framework to capture the asymmetry in the data, which is not possible with the normal distribution.

## Parametrization

The skew-normal distribution has 2 alternative parameterizations. In terms of $\mu$, $\sigma$ and $\alpha$, or in terms of $\mu$, $\tau$ and $\alpha$. 
Where $\mu$ is the location parameter, $\sigma$ is the scale parameter, $\tau$ is the precision parameter, and $\alpha$ is the skewness parameter.

The link between the 2 alternatives is given by:

$$
\begin{align*}
\tau & = \frac{1}{\sigma^2}
\end{align*}
$$

## Probability Density Function (PDF):

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$, $\sigma$, and $\alpha$
:sync: mu-sigma-alpha
```{jupyter-execute}
:hide-code:

from preliz import SkewNormal, style
style.use('preliz-doc')
alphas = [-6., 0., 6.]

for alpha in alphas:
    SkewNormal(mu=0, sigma=1, alpha=alpha).plot_pdf()
```
:::::

:::::{tab-item} Parameters $\mu$, $\tau$, and $\alpha$
:sync: mu-tau-alpha

```{jupyter-execute}
:hide-code:

for alpha in alphas:
    SkewNormal(mu=0, tau=1, alpha=alpha).plot_pdf()
```
:::::
::::::

## Cumulative Distribution Function (CDF):

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$, $\sigma$, and $\alpha$
:sync: mu-sigma-alpha
```{jupyter-execute}
:hide-code:

for alpha in alphas:
    SkewNormal(mu=0, sigma=1, alpha=alpha).plot_cdf()
```
:::::

:::::{tab-item} Parameters $\mu$, $\tau$, and $\alpha$
:sync: mu-tau-alpha

```{jupyter-execute}
:hide-code:

for alpha in alphas:
    SkewNormal(mu=0, tau=1, alpha=alpha).plot_cdf()
```
:::::
::::::

## Key Properties and Parameters:

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu + \sigma \sqrt{\frac{2}{\pi}} \frac{\alpha }{{\sqrt {1+\alpha ^{2}}}}`
Variance  :math:`\sigma^2 \left(  1-\frac{2\alpha^2}{(\alpha^2+1) \pi} \right)`
========  ==========================================
```

**Probability Density Function (PDF):**

$$
f(x \mid \mu, \tau, \alpha) = 2 \Phi((x-\mu)\sqrt{\tau}\alpha) \phi(x,\mu,\tau)
$$

where $\Phi$ is the standard normal CDF and $\phi$ is the normal PDF.

**Cumulative Distribution Function (CDF):**

$$
F(x \mid \mu, \sigma, \alpha) = \frac{1}{2} \left( 1 + \text{erf} \left( \frac{x - \mu}{\sigma \sqrt{2}} \right) \right) - 2 T \left( \frac{x - \mu}{\sigma}, \alpha \right)
$$

where $\text{erf}$ is the [error function](https://en.wikipedia.org/wiki/Error_function) and $T$ is the [Owen's T function](https://en.wikipedia.org/wiki/Owen%27s_T_function).

```{seealso}
:class: seealso

**Related Distributions:**

- [Normal Distribution](normal.md) - The parent distribution from which the skew-normal distribution is derived. When $\alpha = 0$, the skew-normal distribution reduces to the normal distribution.
- [Half-Normal Distribution](halfnormal.md) - When $\alpha$ approaches +/- infinity, the skew-normal distribution becomes a half-normal distribution.
```

## References

- [Wikipedia - Skew-normal distribution](https://en.wikipedia.org/wiki/Skew_normal_distribution)
