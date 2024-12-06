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
# Half-Normal Distribution

<audio controls> <source src="../../_static/halfnormal.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Half-Normal distribution is a continuous probability distribution that is derived from the Normal distribution but is restricted to only positive values. It is characterized by a single scale parameter ($\sigma$), which determines the width of the distribution.

In Bayesian statistics, the Half-Normal distribution is commonly used as a prior for scale parameters.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in [0, \infty)`
Mean      :math:`\dfrac{\sigma \sqrt{2}}{\sqrt{\pi}}`
Variance  :math:`\sigma^2 \left(1 - \dfrac{2}{\pi}\right)`
========  ==========================================
```

**Parameters:**

- $\sigma$ : (float) Standard deviation of the distribution, $\sigma > 0$.
- $\tau$ : (float) Precision of the distribution, $\tau > 0$.

**Alternative parametrization**

The Half-Normal distribution has 2 alternative parameterizations. It can be defined in terms of the standard deviation ($\sigma$) or in terms of the precision ($\tau$).

The link between the 2 alternatives is given by:

$$
\begin{align*}
\tau & = \frac{1}{\sigma^2}
\end{align*}
$$

### Probability Density Function (PDF)

$$
f(x|\sigma) = \sqrt{\dfrac{2}{\pi\sigma^2}} \exp\left(-\dfrac{x^2}{2\sigma^2}\right)
$$

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameter $\sigma$
:sync: sigma
```{jupyter-execute}
:hide-code:

from preliz import HalfNormal, style
style.use('preliz-doc')
sigmas = [.4, 1., 2.]

for sigma in sigmas:
    HalfNormal(sigma).plot_pdf(support=(0, 5))
```
:::::

:::::{tab-item} Parameter $\tau$
:sync: tau

```{jupyter-execute}
:hide-code:

taus = [6.25, 1., 0.25]
for tau in taus:
    HalfNormal(tau=tau).plot_pdf(support=(0, 5))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x|\sigma) = \text{erf}\left(\dfrac{x}{\sigma\sqrt{2}}\right)
$$

where erf is the [error function](https://en.wikipedia.org/wiki/Error_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameter $\sigma$
:sync: sigma
```{jupyter-execute}
:hide-code:

for sigma in sigmas:
    HalfNormal(sigma).plot_cdf(support=(0, 5))
```
:::::

:::::{tab-item} Parameter $\tau$
:sync: tau

```{jupyter-execute}
:hide-code:

for tau in taus:
    HalfNormal(tau=tau).plot_cdf(support=(0, 5))
```
:::::
::::::

```{seealso}
:class: seealso

**Common Alternatives:**

- [Half-Cauchy](halfcauchy.md) - A distribution with heavier tails that considers only the positive half of the Cauchy distribution.

**Related Distributions:**

- [Normal](normal.md) - The parent distribution from which the Half-Normal is derived.
- [Half-Student's t](halfstudentt.md) - As $\nu \to \infty$, the Half-Student's t-distribution converges to the Half-Normal distribution.
- [Truncated Normal](truncated_normal.md) - A Half-Normal distribution can be considered a special case of the Truncated Normal distribution with mean $0$ and lower bound $0$.
```

## References

- [Wikipedia - Half-Normal](https://en.wikipedia.org/wiki/Half-normal_distribution)