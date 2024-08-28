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
# Bernoulli Distribution

<audio controls> <source src="../../_static/bernoulli.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Bernoulli distribution is a discrete probability distribution. It can be thought of as a model for the set of possible outcomes of any single experiment that asks a yes–no question. More formally, it is a random variable which takes the value 1 with probability $p$ and the value 0 with probability $q = 1 − p$.

## Parametrization

The Bernoulli distribution has 2 alternative parametrizations. In terms of $p$ or $\text{logit}(p)$.

The link between the 2 alternatives is given by

$$
\text{logit}(p) = \log(\frac{p}{1-p})
$$


## Probability Density Function (PDF):

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $p$
:sync: p
```{jupyter-execute}
:hide-code:

from preliz import Bernoulli, style
style.use('preliz-doc')
for p in [0.01, 0.5, 0.8]:
    Bernoulli(p).plot_pdf()
```
:::::

:::::{tab-item} Parameters $\text{logit}(p)$ 
:sync: logit_p

```{jupyter-execute}
:hide-code:
for logit_p in [-4.6, 0, 1.38]:
    Bernoulli(logit_p=logit_p).plot_pdf()
```
:::::
::::::

## Cumulative Distribution Function (CDF):

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $p$
:sync: p

```{jupyter-execute}
:hide-code:
for p in [0.01, 0.5, 0.8]:
    Bernoulli(p).plot_cdf()
```
:::::

:::::{tab-item} Parameters $\text{logit}(p)$ 
:sync: logit_p

```{jupyter-execute}
:hide-code:
for logit_p in [-4.6, 0, 1.38]:
    Bernoulli(logit_p=logit_p).plot_cdf()
```
:::::
::::::


## Key properties and parameters:

```{eval-rst}
========  ======================
Support   :math:`x \in \{0, 1\}`
Mean      :math:`p`
Variance  :math:`p (1 - p)`
========  ======================
```

**Probability Density Function (PDF):**

$$
f(x \mid p) = p^{x} (1-p)^{1-x}
$$

**Cumulative Distribution Function (CDF):**

$$
F(x \mid p) = \begin{cases}
    0 & \text{if } k < 0 \\
    1 - p & \text{if } 0 \leq k < 1 \\
    1 & \text{if } k \geq 1
    \end{cases}
$$


```{seealso}
:class: seealso


**Related Distributions:**

- [Binomial](binomial.md) - The Bernoulli distribution is a special case of the Binomial distribution with $N=1$.
- [Categorical](categorical.md) - A generalization of the Bernoulli distribution to more than two outcomes.
```

## References

- [Wikipedia - Bernoulli](https://en.wikipedia.org/wiki/Bernoulli_distribution)





