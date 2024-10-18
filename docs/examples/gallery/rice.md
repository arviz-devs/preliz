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
# Rice Distribution

<audio controls> <source src="../../_static/rice.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Rice distribution is the probability distribution of the magnitude of a circularly-symmetric bivariate normal random variable. It's characterized by two parameters: $v$, which represents the non-centrality parameter, and $\sigma$, the scale parameter. 

The Rice distribution is often used in signal processing, particularly in the analysis of noisy signals, such as radar and communication systems.

## Parametrization

The Rice distribution has two alternative parameterizations: in terms of $v$ and $\sigma$, or in terms of $b$ and $\sigma$. The relationship between the two is given by:

$$
\begin{align*}
b & = \frac{v}{\sigma}
\end{align*}
$$

## Probability Density Function (PDF):

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $v$ and $\sigma$
:sync: v-sigma
```{jupyter-execute}
:hide-code:

from preliz import Rice, style
style.use('preliz-doc')
nus = [0., 0., 4.]
sigmas = [1., 2., 2.]
for nu, sigma in  zip(nus, sigmas):
    Rice(nu, sigma).plot_pdf(support=(0,10))
```
:::::

:::::{tab-item} Parameters $b$ and $\sigma$
:sync: b-sigma

```{jupyter-execute}
:hide-code:

bs = [0., 0., 2.]
for b, sigma in zip(bs, sigmas):
    Rice(b=b, sigma=sigma).plot_pdf(support=(0,10))
```
:::::
::::::

## Cumulative Distribution Function (CDF):

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $v$ and $\sigma$
:sync: v-sigma
```{jupyter-execute}
:hide-code:

for nu, sigma in  zip(nus, sigmas):
    Rice(nu, sigma).plot_cdf(support=(0,10))
```
:::::

:::::{tab-item} Parameters $b$ and $\sigma$
:sync: b-sigma

```{jupyter-execute}
:hide-code:

for b, sigma in zip(bs, sigmas):
    Rice(b=b, sigma=sigma).plot_cdf(support=(0,10))
```
:::::
::::::

## Key properties and parameters:

```{eval-rst}
========  ==============================================================
Support   :math:`x \in (0, \infty)`
Mean      :math:`\sigma \sqrt{\pi /2} L_{1/2}(-\nu^2 / 2\sigma^2)`
Variance  :math:`2\sigma^2 + \nu^2 - \frac{\pi \sigma^2}{2}`
          :math:`L_{1/2}^2\left(\frac{-\nu^2}{2\sigma^2}\right)`
========  ==============================================================
```

**Probability Density Function (PDF):**

$$
f(x|\nu, \sigma) = \frac{x}{\sigma^2} \exp\left(-\frac{x^2 + \nu^2}{2\sigma^2}\right) I_0\left(\frac{x\nu}{\sigma^2}\right)
$$

where $I_0$ is the modified [Bessel function](https://en.wikipedia.org/wiki/Bessel_function) of the first kind.

**Cumulative Distribution Function (CDF):**

$$
F(x|\nu, \sigma) = 1 - Q_1\left(\frac{x}{\sigma}, \frac{\nu}{\sigma}\right)
$$

where $Q_1$ is the [Marcum Q-function](https://en.wikipedia.org/wiki/Marcum_Q-function).

```{seealso}
:class: seealso

**Related Distributions:**

- [Normal](normal.md) - The Rice distribution is the magnitude of a bivariate normal distribution.
```

## References

- Wikipedia - [Rice distribution](https://en.wikipedia.org/wiki/Rice_distribution)
