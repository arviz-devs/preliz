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

# Skew-Student's t Distribution

<audio controls> <source src="../../_static/skewstudentt.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Asymmetric](../../gallery_tags.rst#asymmetric), [Unbounded](../../gallery_tags.rst#unbounded), [Heavy-tailed](../../gallery_tags.rst#heavy-tailed)

The skew-Student's t distribution is a continuous probability distribution that generalizes the Student's t distribution by allowing for non-zero skewness. It is used in various fields when data is skewed and has heavy tails. 

There are several definitions of the skew-Student's t distribution. Here, we use the definition proposed by Jones and Faddy (2003) that is defined in terms of four parameters: the location parameter ($\mu$), the scale parameter ($\sigma$), and two shape parameters ($a$ and $b$). 

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in \mathbb{R}`
Mean      :math:`\mu + \sigma\frac{(a-b) \sqrt{(a+b)}}{2}\frac{\Gamma\left  (a-\frac{1}{2}\right)\Gamma\left(b-\frac{1}{2}\right)}{\Gamma(a) \Gamma(b)}`
Variance  :math:`\sigma^2\left(1 + \frac{\Gamma\left(a-\frac{1}{2}\right)\Gamma\left(b-\frac{1}{2}\right)}{\Gamma(a)\Gamma(b)} - \left(\frac{\Gamma\left(a-\frac{1}{2}\right)\Gamma\left(b-\frac{1}{2}\right)}{\Gamma(a)\Gamma(b)}\right)^2\right)`
========  ==========================================
```

**Parameters:**

- $\mu$ : (float) Location parameter.
- $\sigma$ : (float) Scale parameter.
- $a$ : (float) Shape parameter.
- $b$ : (float) Shape parameter.

**Alternative parametrization**

The Skew-Student's t distribution has two alternative parameterizations. In terms of $\mu$, $\sigma$, $a$ and $b$, or in terms of $\mu$, $\lambda$, $a$ and $b$. 

The link between the two parameterizations is given by:

$$
\lambda = \frac{1}{\sigma^2}
$$

If $a > b$, the skew-Student's t distribution is positively skewed (skewed to the right). If $a < b$, the skew-Student's t distribution is negatively skewed (skewed to the left). If $a = b$, the skew-Student's t distribution reduces to the Student's t distribution with $\nu = 2a$.

The parameter $\sigma$ convergences to the standard deviation and $\lambda$ converges to the precision as $a$ and $b$ approach close, and the value of $a$ gets larger.

### Probability Density Function (PDF)

$$
f(t)=f(t \mid a, b)=C_{a, b}^{-1}\left\{1+\frac{t}{\left(a+b+t^2\right)^{1 / 2}}\right\}^{a+1 / 2}\left\{1-\frac{t}{\left(a+b+t^2\right)^{1 / 2}}\right\}^{b+1 / 2}
$$

Where $C_{a, b}$ is the normalizing constant given by:

$$
C_{a, b}=2^{a+b-1} B(a, b)(a+b)^{1 / 2}
$$

and $B(a, b)$ is the [Beta function](https://en.wikipedia.org/wiki/Beta_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$, $\sigma$, $a$, and $b$
:sync: mu-sigma-a-b
```{jupyter-execute}
:hide-code:

from preliz import SkewStudentT, style
style.use('preliz-doc')
mus = [0., 0., 0., 0.]
sigmas = [1., 1., 1., 2.]
a_s = [1., 1., 3., 1.]
b_s = [1., 3., 1., 3.]
for mu, sigma, a, b in zip(mus, sigmas, a_s, b_s):
    SkewStudentT(mu, sigma, a, b).plot_pdf(support=(-10,10))
```
:::::

:::::{tab-item} Parameters $\mu$, $\lambda$, $a$, and $b$
:sync: mu-lambda-a-b
```{jupyter-execute}
:hide-code:

lambdas = [1., 1., 1., 0.25]
for mu, lam, a, b in zip(mus, lambdas, a_s, b_s):
    SkewStudentT(mu, lam=lam, a=a, b=b).plot_pdf(support=(-10,10))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(t \mid a, b) = I_{\frac{1+t/\sqrt{a + b + t^2}}{2}}(a, b)
$$

where $I_x(a, b)$ is the [regularized incomplete beta function](https://en.wikipedia.org/wiki/Beta_function#Incomplete_beta_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$, $\sigma$, $a$, and $b$
:sync: mu-sigma-a-b
```{jupyter-execute}
:hide-code:

for mu, sigma, a, b in zip(mus, sigmas, a_s, b_s):
    SkewStudentT(mu, sigma, a, b).plot_cdf(support=(-10,10))
```
:::::

:::::{tab-item} Parameters $\mu$, $\lambda$, $a$, and $b$
:sync: mu-lambda-a-b
```{jupyter-execute}
:hide-code:

for mu, lam, a, b in zip(mus, lambdas, a_s, b_s):
    SkewStudentT(mu, lam=lam, a=a, b=b).plot_cdf(support=(-10,10))
```
:::::
::::::

```{seealso}
:class: seealso

**Common Alternatives:**

- [Student's t Distribution](students_t.md) - The parent distribution of the skew-Student's t distribution. Less flexible than the skew-Student's t distribution.

**Related Distributions:**

- [Skew-Normal Distribution](skewnormal.md) - A related distribution that generalizes the normal distribution by introducing a skewness parameter, but has lighter tails than the skew-Student's t distribution.
```

## References:

- M.C. Jones and M.J. Faddy. “A skew extension of the t distribution, with applications” Journal of the Royal Statistical Society, Series B (Statistical Methodology) 65, no. 1 (2003): 159-174. [DOI:10.1111/1467-9868.00378](https://doi.org/10.1111/1467-9868.00378)