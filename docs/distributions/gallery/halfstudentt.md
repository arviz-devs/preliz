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
# Half-Student's t Distribution

<audio controls> <source src="../../_static/halfstudentt.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Non-Negative](../../gallery_tags.rst#non-negative), [Asymmetric](../../gallery_tags.rst#asymmetric), [Heavy-tailed](../../gallery_tags.rst#heavy-tailed)

The Half-Student's t distribution, also known as the half-t distribution, is a continuous probability distribution that is derived from the Student's t distribution but is restricted to only positive values. It is characterized by two parameters: the degrees of freedom ($\nu$) and the scale parameter ($\sigma$), which determines the width of the distribution. The smaller the value of $\nu$, the heavier the tails of the distribution.

In Bayesian statistics, the Half-Student's t distribution is often used as a prior for scale parameters.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in [0, \infty)`
Mean      .. math::
                2\sigma\sqrt{\frac{\nu}{\pi}}\
                \frac{\Gamma\left(\frac{\nu+1}{2}\right)}
                {\Gamma\left(\frac{\nu}{2}\right)(\nu-1)}\, \text{for } \nu > 2
Variance  .. math::
                \sigma^2\left(\frac{\nu}{\nu - 2}-\
                \frac{4\nu}{\pi(\nu-1)^2}\left(\frac{\Gamma\left(\frac{\nu+1}{2}\right)}
                {\Gamma\left(\frac{\nu}{2}\right)}\right)^2\right) \text{for } \nu > 2\, \infty\
                \text{for } 1 < \nu \le 2\, \text{otherwise undefined}
========  ==========================================
```

**Parameters:**

- $\nu$ : (float) Degrees of freedom.
- $\sigma$ : (float) Scale parameter.
- $\lambda$ : (float) Precision parameter.

**Alternative parametrization**

The Half-Student's t distribution has 2 alternative parameterizations. In terms of $\nu$ and $\sigma$, or in terms of $\nu$ and $\lambda$.

The link between the 2 alternatives is given by:

$$
\lambda = \frac{1}{\sigma^2}
$$

where $\sigma$ is the standard deviation as $\nu$ increases, and $\lambda$ is the precision as $\nu$ increases.

### Probability Density Function (PDF)

$$
f(x \mid \sigma,\nu) =
    \frac{2\;\Gamma\left(\frac{\nu+1}{2}\right)}
    {\Gamma\left(\frac{\nu}{2}\right)\sqrt{\nu\pi\sigma^2}}
    \left(1+\frac{1}{\nu}\frac{x^2}{\sigma^2}\right)^{-\frac{\nu+1}{2}}
$$

where $\Gamma$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\nu$ and $\sigma$
:sync: nu_sigma
```{jupyter-execute}
:hide-code:

from preliz import HalfStudentT, style
style.use('preliz-doc')
nus = [2., 5., 5.]
sigmas = [1., 1., 2.]

for nu, sigma in zip(nus, sigmas):
    HalfStudentT(nu, sigma).plot_pdf(support=(0, 5))
```
:::::

:::::{tab-item} Parameters $\nu$ and $\lambda$
:sync: nu_lambda

```{jupyter-execute}
:hide-code:

lambdas = [1., 1., 0.25]
for nu, lam in zip(nus, lambdas):
    HalfStudentT(nu, lam=lam).plot_pdf(support=(0, 5))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x \mid \sigma,\nu) = 
\begin{cases} 
\frac{1}{2} \cdot I_{\frac{\nu}{x^2 + \nu}}\left(\frac{\nu}{2}, \frac{1}{2}\right) & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}
$$

where $I_x(a, b)$ denotes the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\nu$ and $\sigma$
:sync: nu_sigma
```{jupyter-execute}
:hide-code:

for nu, sigma in zip(nus, sigmas):
    HalfStudentT(nu, sigma).plot_cdf(support=(0, 5))
```
:::::

:::::{tab-item} Parameters $\nu$ and $\lambda$
:sync: nu_lambda

```{jupyter-execute}
:hide-code:

for nu, lam in zip(nus, lambdas):
    HalfStudentT(nu, lam=lam).plot_cdf(support=(0, 5))
```
:::::
::::::

```{seealso}
:class: seealso

**Common Alternatives:**

- [Half-Cauchy](halfcauchy.md) - The Half-Cauchy distribution is a special case of the Half-Student's t distribution with $\nu = 1$.
- [Half-Normal](halfnormal.md) - As $\nu \to \infty$, the Half-Student's t distribution approaches the Half-Normal distribution.

**Related Distributions:**

- [Student's t](studentt.md) - The Student's t distribution is the parent distribution from which the Half-Student's t distribution is derived.
```

## References

- [Wikipedia - Folded-t and Half-t Distributions](https://en.wikipedia.org/wiki/Folded-t_and_half-t_distributions)
- [Wikipedia - Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)