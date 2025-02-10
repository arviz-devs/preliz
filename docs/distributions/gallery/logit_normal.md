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
# Logit-Normal Distribution

<audio controls> <source src="../../_static/logitnormal.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Univariate](../../gallery_tags.rst#univariate), [Continuous](../../gallery_tags.rst#continuous), [Asymmetric](../../gallery_tags.rst#asymmetric), [Non-Negative](../../gallery_tags.rst#non-negative), [Heavy-tailed](../../gallery_tags.rst#heavy-tailed)

The logit-normal distribution is a continuous probability distribution of a random variable whose [logit](https://en.wikipedia.org/wiki/Logit) (or log-odds) is normally distributed. Thus, if a random variable $X$ follows a logit-normal distribution, then $ Y = \text{logit}(X) = \log\left(\frac{X}{1-X}\right)$ is normally distributed. It is defined for values of $x$ between 0 and 1. It is characterized by two parameters: $\mu$ and $\sigma$, which are the mean and standard deviation of the logit-transformed variable, respectively, not the original variable.

The logit-normal distribution is useful in modeling proportions or ratios. 

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in (0, 1)`
Mean      no analytical solution
Variance  no analytical solution
========  ==========================================
```

**Parameters:**

- $\mu$ : (float) The mean of the logit-transformed variable.
- $\sigma$ : (float) The standard deviation of the logit-transformed variable.
- $\tau$ : (float) The precision of the logit-transformed variable, $\tau = \frac{1}{\sigma^2}$.

**Alternative parametrization**

The logit-normal distribution can be parametrized in terms of $\mu$ and $\sigma$ or in terms of $\mu$ and $\tau$.

The link between the two parametrizations is given by:

$$
\tau = \frac{1}{\sigma^2}
$$

### Probability Density Function (PDF)

$$
f(x \mid \mu, \sigma) = \frac{1}{x(1-x)\sigma\sqrt{2\pi}}\exp\left(-\frac{1}{2}\left(\frac{\text{logit}(x)-\mu}{\sigma}\right)^2\right)
$$

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$ and $\sigma$
:sync: mu-sigma
```{jupyter-execute}
:hide-code:

from preliz import LogitNormal, style
style.use('preliz-doc')
mus = [0., 0., 0., 1.]
sigmas = [0.3, 1., 2., 1.]
for mu, sigma in zip(mus, sigmas):
    LogitNormal(mu, sigma).plot_pdf()
```
:::::

:::::{tab-item} Parameters $\mu$ and $\tau$
:sync: mu-tau

```{jupyter-execute}
:hide-code:

taus = [11.11, 1., 0.25, 1.]
for mu, tau in zip(mus, taus):
    LogitNormal(mu, tau=tau).plot_pdf()
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x \mid \mu, \sigma) = \frac{1}{2} + \frac{1}{2}\text{erf}\left(\frac{\text{logit}(x)-\mu}{\sigma\sqrt{2}}\right)
$$

where erf is the [error function](https://en.wikipedia.org/wiki/Error_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\mu$ and $\sigma$
:sync: mu-sigma
```{jupyter-execute}
:hide-code:

for mu, sigma in zip(mus, sigmas):
    LogitNormal(mu, sigma).plot_cdf()
```
:::::

:::::{tab-item} Parameters $\mu$ and $\tau$
:sync: mu-tau
```{jupyter-execute}
:hide-code:

for mu, tau in zip(mus, taus):
    LogitNormal(mu, tau=tau).plot_cdf()
```
:::::
::::::

```{seealso}
:class: seealso

**Related Distributions:**

- [Normal Distribution](normal.md) - The logit-normal distribution is directly related to the normal distribution since if a variable is logit-normally distributed, its logit follows a normal distribution. This relationship is crucial for understanding the logit-normal's properties and applications.
```

## References

- [Wikipedia - Logit-normal distribution](https://en.wikipedia.org/wiki/Logit-normal_distribution)