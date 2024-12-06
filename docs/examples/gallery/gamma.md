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
# Gamma Distribution

<audio controls> <source src="../../_static/gamma.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The gamma distribution is a continuous probability distribution that describes the time until a specified number of events occur in a Poisson process, in which $\alpha$ events occur continuously and independently at a constant average rate $\beta$. The gamma distribution is characterized by two parameters: the shape parameter $\alpha$ and the rate parameter $\beta$.

The gamma distribution is widely used in many settings, including modeling the time until a specified number of decays of a radioactive atom, the time until a specified number of seizures in patients with epilepsy, the time until a specified number of failures of a machine, and the time until a specified number of changes in the price of a stock. 

The gamma distribution is used as a conjugate prior for the exponential distribution and for the precision of the normal distribution. It can also used as prior for scale parameters, in particular when there is interest in avoiding values close to zero or one wants to include more information than with other prior commonly used for scale parameter like the halfnormal or the exponential.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in (0, \infty)`
Mean      :math:`\frac{\alpha}{\beta}`
Variance  :math:`\frac{\alpha}{\beta^2}`
========  ==========================================
```

**Parameters:**

- $\alpha$ : (float) Shape parameter, $\alpha > 0$.
- $\beta$ : (float) Rate parameter, $\beta > 0$.
- $\mu$ : (float) Mean of the distribution.
- $\sigma$ : (float) Standard deviation of the distribution.

**Alternative parametrization**

The gamma distribution has two alternative parameterizations. In terms of $\alpha$ and $\beta$, and $\mu$ and $\sigma$.

The link between the 2 alternatives is given by:

$$
\begin{align*}
\alpha & = \frac{\mu^2}{\sigma^2} \\
\beta & = \frac{\mu}{\sigma^2}
\end{align*}
$$

where $\mu$ is the mean of the distribution and $\sigma$ is the standard deviation.

### Probability Density Function (PDF)

$$
f(x|\alpha, \beta) = \frac{\beta^\alpha x^{\alpha - 1} e^{-\beta x}}{\Gamma(\alpha)}
$$

where $\Gamma$ is the [gamma function](https://en.wikipedia.org/wiki/Gamma_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\alpha$ and $\beta$
:sync: alpha-beta
```{jupyter-execute}
:hide-code:

from preliz import Gamma, style
style.use('preliz-doc')
alphas = [1, 3, 7.5]
betas = [0.5, 1., 1.]

for alpha, beta in zip(alphas, betas):
    Gamma(alpha, beta).plot_pdf()
```
:::::

:::::{tab-item} Parameters $\mu$ and $\sigma$
:sync: mu-sigma

```{jupyter-execute}
:hide-code:

mus = [2, 3, 7.5]
sigmas = [2, 1.732, 2.739]


for mu, sigma in zip(mus, sigmas):
    Gamma(mu=mu, sigma=sigma).plot_pdf()
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x|\alpha, \beta) = \frac{1}{\Gamma(\alpha)} \gamma(\alpha, \beta x)
$$

where $\gamma$ is the [lower incomplete gamma function](https://en.wikipedia.org/wiki/Incomplete_gamma_function).

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\alpha$ and $\beta$
:sync: alpha-beta
```{jupyter-execute}
:hide-code:

for alpha, beta in zip(alphas, betas):
    Gamma(alpha, beta).plot_cdf()
```
:::::

:::::{tab-item} Parameters $\mu$ and $\sigma$
:sync: mu-sigma

```{jupyter-execute}
:hide-code:

for mu, sigma in zip(mus, sigmas):
    Gamma(mu=mu, sigma=sigma).plot_cdf()
```
:::::
::::::

```{seealso}
:class: seealso

**Common Alternatives:**

- [Exponential Distribution](exponential.md) - A special case of the Gamma distribution with the shape parameter $\alpha = 1$.
- [Chi-Squared Distribution](chisquared.md) - A special case of the Gamma distribution with the shape parameter $\alpha = \nu/2$ and the scale parameter $\beta = 1/2$.

**Related Distributions:**

- [Inverse Gamma Distribution](inversegamma.md) - The reciprocal of a gamma-distributed random variable.
- [Poisson Distribution](poisson.md) - While the Gamma distribution models the time until a specified number of events occur in a Poisson process, the Poisson distribution models the number of events in fixed intervals of time or space.
- [Negative Binomial Distribution](negative-binomial.md) - The discrete counterpart of the Gamma distribution, modeling the number of trials needed to achieve a specified number of successes in a sequence of Bernoulli trials.
```

## References

- [Wikipedia - Gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution)