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
# Zero-Inflated Negative Binomial Distribution

<audio controls> <source src="../../_static/zeroinflatednegativebinomial.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

The Zero-Inflated Negative Binomial (ZINB) distribution is a discrete probability distribution used to model count data characterized by an excess of zeros. It combines two components: a NegativeBinomial component, which describe count values and a zero-inflation component, which accounts for the extra zeros.

## Key properties and parameters

```{eval-rst}
========  ==========================
Support   :math:`x \in \mathbb{N}_0`
Mean      :math:`\psi\mu`
Variance  :math:`\psi \left(\frac{{\mu^2}}{{\alpha}}\right) + \psi \mu + \psi \mu^2 - \psi^2 \mu^2`
========  ==========================
```

**Parameters:**

- $\psi$ : (float) Expected proportion of Negative Binomial variates, $0 \leq \psi \leq 1$.
- $\mu$ : (float) Poisson distribution mean parameter, $\mu > 0$.
- $\alpha$ : (float) Gamma distribution shape parameter, $\alpha > 0$.
- $n$ : (int) Number of target success trials, $n \geq 0$.
- $p$ : (float) Probability of success in each trial, $0 \leq p \leq 1$.

**Alternative Parameterization:**

The ZINB distribution can be parametrized either in terms of $\psi$, $\mu$ and $\alpha$ or in terms of $\psi$, $n$ and $p$. The link between the two parameterizations is given by:

$$
\begin{align*}
\mu = \frac{n(1-p)}{p} \\
\alpha = n
\end{align*}
$$

### Probability Mass Function (PMF)

$$
f(x \mid \psi, \mu, \alpha) = \left\{
    \begin{array}{l}
    (1-\psi) + \psi \left (
        \frac{\alpha}{\alpha+\mu}
    \right) ^\alpha, \text{if } x = 0 \\
    \psi \frac{\Gamma(x+\alpha)}{x! \Gamma(\alpha)} \left (
        \frac{\alpha}{\mu+\alpha}
    \right)^\alpha \left(
        \frac{\mu}{\mu+\alpha}
    \right)^x, \text{if } x=1,2,3,\ldots
    \end{array}
\right.
$$
    
::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\psi$, $\mu$ and $\alpha$
:sync: psi-mu-alpha

```{jupyter-execute}
:hide-code:

from preliz import ZeroInflatedNegativeBinomial, style
style.use('preliz-doc')
psis = [0.7, 0.7]
mus = [2, 8]
alphas = [2, 4]
for psi, mu, alpha in zip(psis, mus, alphas):
    ZeroInflatedNegativeBinomial(psi, mu=mu, alpha=alpha).plot_pdf(support=(0,25))
```
:::::

:::::{tab-item} Parameters $\psi$, $n$ and $p$
:sync: psi-n-p

```{jupyter-execute}
:hide-code:

ns = [2, 4]
ps = [0.5, 0.33]
for psi, n, p in zip(psis, ns, ps):
    ZeroInflatedNegativeBinomial(psi, n=n, p=p).plot_pdf(support=(0,25))
```
:::::
::::::

### Cumulative Distribution Function (CDF)

The CDF of the Zero-Inflated Negative Binomial distribution is given by:

$$
F(x \mid \psi, \mu, \alpha) = \sum_{i=0}^{x} f(i \mid \psi, \mu, \alpha)
$$

::::::{tab-set}
:class: full-width

:::::{tab-item} Parameters $\psi$, $\mu$ and $\alpha$
:sync: psi-mu-alpha

```{jupyter-execute}
:hide-code:

for psi, mu, alpha in zip(psis, mus, alphas):
    ZeroInflatedNegativeBinomial(psi, mu=mu, alpha=alpha).plot_cdf(support=(0,25))
```
:::::

:::::{tab-item} Parameters $\psi$, $n$ and $p$
:sync: psi-n-p

```{jupyter-execute}
:hide-code:

for psi, n, p in zip(psis, ns, ps):
    ZeroInflatedNegativeBinomial(psi, n=n, p=p).plot_cdf(support=(0,25))
```
:::::
::::::

```{seealso}

**Common Alternatives:**

- [Zero-Inflated Poisson Distribution](zeroinflatedpoisson.md) - A similar distribution for count data with excess zeros.
- [Negative Binomial Distribution](negativebinomial.md) - A distribution for modeling the number of successes in a sequence of independent and identically distributed Bernoulli trials before a specified number of failures occurs.

**Related Distributions:**

- [Zero-Inflated Binomial Distribution](zeroinflatedbinomial.md) - Combines the Binomial distribution with a zero-inflation component, useful for modeling count data with an excess of zeros.
```

## References

- [Wikipedia - Zero-Inflated Models](https://en.wikipedia.org/wiki/Zero-inflated_model)
- [Wikipedia - Negative Binomial Distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution)