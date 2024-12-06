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
# Exponential Distribution

<audio controls> <source src="../../_static/exponential.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The exponential distribution is a continuous probability distribution that describes the time between events in a Poisson process, in which events occur continuously and independently at a constant average rate. It has a memorylessness property, which means that the probability of the event occurring in the next instant of time does not depend on how much time has already elapsed. It is characterized by the rate parameter $\lambda$ which is the reciprocal of the mean of the distribution.

Many real-world phenomena can be modeled using the exponential distribution across various fields. For example, it is used in physics to model the time between decays of a radioactive atom, in medicine to model the time between seizures in patients with epilepsy, in engineering to model the time between failures of a machine, and in finance to model the time between changes in the price of a stock.

In Bayesian modeling, the exponential distribution is often used as a prior distribution for scale parameters, which must always be positive.

## Key properties and parameters

```{eval-rst}
========  ==========================================
Support   :math:`x \in [0, \infty)`
Mean      :math:`\frac{1}{\lambda}`
Variance  :math:`\frac{1}{\lambda^2}`
========  ==========================================
```

**Parameters:**

- $\lambda$ : (float) Rate parameter, $\lambda > 0$.
- $\beta$ : (float) Scale parameter, $\beta > 0$.

**Alternative parametrization**

The exponential distribution can be parametrized in terms of the rate parameter $\lambda$ or the scale parameter $\beta$.

The link between the 2 alternatives is given by:

$$
\beta = \frac{1}{\lambda}
$$

### Probability Density Function (PDF)

$$
f(x|\lambda) = \lambda e^{-\lambda x}
$$

::::::{tab-set} 
:class: full-width

:::::{tab-item} Parameter $\lambda$
:sync: rate
```{jupyter-execute}
:hide-code:

from preliz import Exponential, style
style.use('preliz-doc')
lambdas = [0.5, 1., 2.]

for lam in lambdas:
    Exponential(lam).plot_pdf(support=(0, 5))
```
:::::

:::::{tab-item} Parameter $\beta$
:sync: scale
```{jupyter-execute}
:hide-code:

betas = [2., 1., 0.5]
for beta in betas:
    Exponential(beta=beta).plot_pdf(support=(0, 5))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

$$
F(x|\lambda) = 1 - e^{-\lambda x}
$$

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameter $\lambda$
:sync: rate
```{jupyter-execute}
:hide-code:

for lam in lambdas:
    Exponential(lam).plot_cdf(support=(0, 5))
```
:::::

:::::{tab-item} Parameter $\beta$
:sync: scale
```{jupyter-execute}
:hide-code:

for beta in betas:
    Exponential(beta=beta).plot_cdf(support=(0, 5))
```
:::::
::::::

```{seealso}
:class: seealso

**Common Alternatives:**

- [Gamma](gamma.md) - The Exponential distribution is a special case of the Gamma distribution with the shape parameter $\alpha = 1$.
- [Weibull](weibull.md) - The Exponential distribution is a special case of the Weibull distribution with the shape parameter $\alpha = 1$.

**Related Distributions:**

- [Poisson](poisson.md) - While the Exponential distribution models the time between events in a Poisson process, the Poisson distribution models the number of events in fixed intervals of time or space.
- [Geometric](geometric.md) - The discrete counterpart of the Exponential distribution, modeling the number of trials needed to achieve the first success in a sequence of Bernoulli trials.
- [Pareto](pareto.md) - If X is exponentially distributed with rate parameter $\lambda$, then $m e^X$ is Pareto distributed with shape parameter $\alpha = \lambda$ and scale parameter $m$.
```

## References

- [Wikipedia - Exponential](https://en.wikipedia.org/wiki/Exponential_distribution)