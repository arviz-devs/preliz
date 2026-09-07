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
# Wishart Distribution
<audio controls> <source src="../../audios/wishart.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Multivariate](../../gallery_tags.rst#multivariate), [Continuous](../../gallery_tags.rst#continuous), [Asymmetric](../../gallery_tags.rst#asymmetric), [Bounded](../../gallery_tags.rst#bounded), [Heavy-tailed](../../gallery_tags.rst#heavy-tailed)

The Wishart distribution is a multivariate, continuous probability distribution over the space of $p \times p$ symmetric, positive semi-definite matrices. It is the generalization to multiple dimensions of the chi-squared distribution: if $X$ is Wishart distributed, it can be thought of as the distribution of the sample covariance matrix obtained from $\nu$ independent draws of a $p$-dimensional zero-mean Gaussian with covariance $V$.

The Wishart distribution is most commonly used as a conjugate prior for the precision matrix (inverse covariance) of a multivariate normal distribution in Bayesian inference.

## Key properties and parameters

```{eval-rst}
========  ==============================================================
Support   :math:`X` a :math:`p \times p` positive semi-definite matrix
Mean      :math:`\nu V`
Mode      :math:`(\nu - p - 1) V, \text{ for } \nu \geq p + 1`
Variance  :math:`\nu (V_{ij}^2 + V_{ii}V_{jj})`
========  ==============================================================
```

**Parameters:**

- $\nu$ : (float) Degrees of freedom, $\nu > p - 1$.
- $V$ : (array_like) Scale matrix, $p \times p$ positive definite.

### Probability Density Function (PDF)

$$
f(X \mid \nu, V) =
    \frac{|X|^{(\nu - p - 1)/2} \exp\left(-\frac{1}{2}\text{tr}(V^{-1}X)\right)}
    {2^{\nu p/2} |V|^{\nu/2} \Gamma_p\left(\frac{\nu}{2}\right)}
$$

```{jupyter-execute}
:hide-code:
from preliz import Wishart, style
style.use('preliz-doc')
nus = [1., 3., 6.]
Vs = [[[1]], [[2]], [[3]]]
for nu, V in zip(nus, Vs):
    Wishart(nu, V).plot_pdf(support=(0, 10))
```

```{seealso}
:class: seealso
**Related Distributions:**
- [Multivariate Normal](multivariatenormal.md) - The Wishart distribution is commonly used as a prior for the precision matrix of a Multivariate Normal distribution.
- [Chi Squared](chisquared.md) - The univariate ($p=1$) special case of the Wishart distribution is a scaled Chi Squared distribution.
```

## References

- Wikipedia - [Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution)
