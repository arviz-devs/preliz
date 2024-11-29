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
# Asymmetric Laplace Distribution

<audio controls> <source src="../../_static/asymmetriclaplace.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Asymmetric Laplace distribution (ALD) is a continuous probability distribution. It is a generalization of the Laplace distribution. The ALD consists of two exponential distributions of unequal scale back to back about $x = \mu$, adjusted to assure continuity and normalization. 

The difference of two variates exponentially distributed with different means and rate parameters will be distributed according to the ALD. When the two rate parameters are equal, the difference will be distributed according to the Laplace distribution.

The ALD, with parameters  $\mu$, $b$ and $q$, is commonly used for performing quantile regression in a Bayesian inference context, with $q$ indicating the desired quantile.

## Key properties and parameters

```{eval-rst}
========  =========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu-\frac{\\\kappa-1/\kappa}b`
Variance  :math:`\frac{1+\kappa^{4}}{b^2\kappa^2 }`
========  =========================================
```

**Parameters:**

- $\kappa$ : (float) Symmetry parameter, $\kappa > 0$.
- $\mu$ : (float) Location parameter.
- $b$ : (float) Scale parameter.
- $q$ : (float) Symmetry parameter, $0 < q < 1$.

**Alternative parametrization**

The ALD has 2 alternative parametrizations. In terms of $\kappa$, $\mu$ and $b$, or $q$, $\mu$ and $b$.

The link between the 2 alternatives is given by

$$
\begin{align*}
\kappa = \sqrt{\frac{1-q}{q}}
\end{align*}
$$

where $\kappa$ and $q$ are symmetry parameters, $\mu$ is the location parameter and $b$ is the scale parameter.

### Probability Density Function (PDF)

$$
{f(x \mid b,\kappa,\mu) =
    \left({\frac{b}{\kappa + 1/\kappa}}\right)\, e^{-(x-\mu) b \,s\kappa ^{s}}}
$$

where s=sgn(x-m), and [sgn](https://en.wikipedia.org/wiki/Sign_function) is the sign function.

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\kappa$, $\mu$ and $b$
:sync: kappa-mu-b

```{jupyter-execute}
:hide-code:

from preliz import AsymmetricLaplace, style
style.use('preliz-doc')
kappas = [1., 2., .5]
mus = [0., 0., 3.]
bs = [1., 1., 1.]
for kappa, mu, b in zip(kappas, mus, bs):
    AsymmetricLaplace(kappa, mu, b).plot_pdf(support=(-10, 10))
```
:::::

:::::{tab-item} Parameters $q$, $\mu$ and $b$
:sync: q-mu-b

```{jupyter-execute}
:hide-code:

qs = [.5, .8, .2]
for q, mu, b in zip(qs, mus, bs):
    AsymmetricLaplace(q=q, mu=mu, b=b).plot_pdf(support=(-10, 10))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x \mid b,\kappa,\mu)  = 
    \begin{cases}
      \frac{\kappa^2}{1+\kappa^2}\exp(b \kappa(x-\mu)) & \text{if } x \leq \mu \\
     1-\frac{1}{1+\kappa^2} \exp(-b \kappa(x-\mu))  & \text{if } x > \mu
    \end{cases}
$$

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\kappa$, $\mu$ and $b$
:sync: kappa-mu-b

```{jupyter-execute}
:hide-code:
for kappa, mu, b in zip(kappas, mus, bs):
    AsymmetricLaplace(kappa, mu, b).plot_cdf(support=(-10, 10))
```
:::::

:::::{tab-item} Parameters $q$, $\mu$ and $b$
:sync: q-mu-b

```{jupyter-execute}
:hide-code:
for q, mu, b in zip(qs, mus, bs):
    AsymmetricLaplace(q=q, mu=mu, b=b).plot_cdf(support=(-10, 10))
```
:::::
::::::

```{seealso}
:class: seealso

**Related Distributions:**
- [Laplace](laplace.md) - The Asymmetric Laplace distribution is a generalization of the Laplace distribution, for the latter, the two scale parameters are equal.
```

## References

- Wikipedia - [AsymmetricLaplace distribution](https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution)