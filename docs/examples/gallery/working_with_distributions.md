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
# Working with Distributions

```{jupyter-execute}
import preliz as pz
```

```{jupyter-execute}

pz.style.use("preliz-doc")
```

PreliZ offers a large variety of distributions, you can check them in the [Gallery](https://preliz.readthedocs.io/en/latest/gallery_content.html). Distributions are implemented as classes, you can instantiate distributions without specifying any parameters. For instance, you can do

```{jupyter-execute}

dist = pz.Normal()
dist
```

Even for unparametrized distributions you can get some general information, like the support.

```{jupyter-execute}

dist.support
```

Notice that for some distributions the support is determined by its parameters, for instance for the `Binomial` distribution the support depends on its parameter `n`.

When we define the parameters of a distribution we will get more interesting properties and methods. Let’s see some of them.

## Properties of Distributions

Once we have set the parameters of a distribution we can obtain a few of its properties. The summary method, returns the mean, median, standard deviation and lower and upper values for the equal-tailed interval (ETI). 

```{jupyter-execute}

dist = pz.Beta(2, 5)
dist.summary()
```
The ETI is the interval that contains a given mass of the distribution, with equal mass in both tails.cFollowing ArviZ, the default mass for these intervals is 0.89. For `pz.summary()` and other functions in PreliZ, you can change it with the argument `mass`.

```{jupyter-execute}

dist.summary(mass=0.7)
```

Additionally, out-of-the-box, we can compute the highest density interval (HDI), which is the shortest interval containing a given mass of the distribution.

```{jupyter-execute}

dist.eti(), dist.hdi()
```

Sometimes we define distributions to then sample from them.

```{jupyter-execute}

dist.rvs(10)
```

Or find out the probability of getting a value below some value of interest. For isntance, 0.265. For that we can call the cumulative distribution function.

```{jupyter-execute}

dist.cdf(0.265)
```

or compute the quantiles

```{jupyter-execute}

dist.ppf([0.1, 0.5, 0.9])
```

`ppf` stands for percentile point function. Here we follow the SciPy nomenclature. Perhaps a more common name for this function is the quantile function or inverse cumulative distribution function.

PreliZ distributions offer many other methods like the moments: mean, variance, skewness, and kurtosis. We can get them one by one

```{jupyter-execute}

dist.mean(), dist.var(), dist.skewness(), dist.kurtosis()
```

or in a single call

```{jupyter-execute}

dist.moments()
```

Actually, `moments` suports a specifying the moment with the “m” (mean), “v”, (variance), “s” (skewness), and “k” (kurtosis). Additionally you can ask for the standard deviation with “d”. So “md” will return the mean and standard deviation.

```{jupyter-execute}

dist.moments(types="md")
```

One important note is that PreliZ computes the [excess kurtosis](https://en.wikipedia.org/wiki/Kurtosis#Excess_kurtosis), then for a Normal we get 0.

```{jupyter-execute}

pz.Normal(0, 1).kurtosis()
```

## Alternative parametrization

Many distributions in PreliZ can be defined using different sets of parameters. For instance, we can define the a Gamma distribution in terms or `alpha`, and `beta` or in terms of `mu` and `sigma`. The following code and figures show how to generate the same Gamma distribution using both parametrizations.

```{jupyter-execute}

dist = pz.Gamma(mu=2, sigma=1)
dist.summary()
```

Now we can create the same distribution using the `alpha` and `beta` parameters.

```{jupyter-execute}
pz.Gamma(dist.alpha, dist.beta).summary() 
```

## Visualizing distributions

Numerical values and summaries are useful, but often we can gain a lot of understanding by visual inspection. For instance, when setting priors for Bayesian models, sometimes all we need to do is to quickly inspect the *shape* of a distribution. We usually want to identify where the bulk of the mass is, or how thick the tails are.

With PreliZ we can easily plot the [probability density function](https://en.wikipedia.org/wiki/Probability_density_function) (pdf) of a given distribution.

```{jupyter-execute}

pz.Beta(2, 5).plot_pdf(pointinterval=True)
```

We get the PDF in blue, and because we passed the argument `pointinterval=True` we also get a box-plot-like element at the bottom. The white dot represents the median. If `interval='hdi'` (the default), the thicker line is the HDI 50% and the thin line is the HDI 94%.

Alternatively, you can set `interval='eti'`, then the thicker line would be the interquartile range, which is defined by the quantiles 0.25 and 0.75 and then represents the central 50% of the distribution. The thinner line would be the interval defined by the 0.03 and 0.97 quantiles.

Discrete distributions don’t have PDF, they have PMF. But for simplicity in preliz we always use the `pdf` to refer to both of them. Then for a discrete distribution we also have a `plot_pdf` method, instead of `plot_pmf`.

```{jupyter-execute}

pz.Poisson(4.3).plot_pdf();
```

We can include more than one distribution in the same plot. This can be useful to compare the impact of parameters on a given distribution or even different distributions.

```{jupyter-execute}

pz.Beta(2, 5).plot_pdf()
pz.Beta(5, 2).plot_pdf()
```

Other functions that are easy to plot are the [cumulative distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function).

```{jupyter-execute}

pz.Gamma(2, 0.5).plot_cdf()
pz.Gamma(2, 1).plot_cdf()
```

Other functions that can be plotted are the inverse of the CDF, `plot_ppf` (also known as percent point function), survival function (`plot_sf`), and the inverse survival function (`plot_isf`).

Alternatively, we can use the top-level `pz.plot()` function to plot any of these functions. 

```{jupyter-execute}
pz.plot(pz.Gamma(2, 0.5));
```

The default is to plot the PDF. But we can plot any other function as well. For instance, for the survival function:

```{jupyter-execute}
pz.plot(pz.Gamma(2, 0.5), kind="sf");
```


If we are not very familiar with a distribution, we may want to explore how the parameters affects the “shape” of the distribution. This could be easier to do interactively.

```{jupyter-execute}

pz.Gamma(mu=2, sigma=1).plot_interactive()
```

As a general rule, PreliZ distributions do not have default values for their parameters, as we saw at the beginning of this notebook for the Normal. Nevertheless, the `plot_interactive()` method provides a default initialization, which can be handy if we are not very familiar with the parameters a distribution can take.

```{jupyter-execute}

pz.Gamma().plot_interactive()
```

The slider will conform to the bounds of the parameters, for instance the `alpha` parameter of the Gamma distribution is restricted to be positive. So the slider for alpha will also be restricted to positive values. Because there is no upper bound, PreliZ will pick one for you, if you want to try with higher values of the parameter just initialize the distribution at higher values, like

```{jupyter-execute}

pz.Gamma(alpha=50, beta=1).plot_interactive()
```

## Distributions modifiers

PreliZ supports some special “distributions” that act as modifiers of other univariate distributions. Currently, we have:

- [Censored](https://preliz.readthedocs.io/en/latest/distributions/gallery/censored.html) - A censored distribution arises when data is limited to a certain range, and values outside this range are not recorded.

- [Truncated](https://preliz.readthedocs.io/en/latest/distributions/gallery/truncated.html) - Truncated distributions arise in cases where the ability to record, or even to know about, occurrences is limited to values which lie above or below a given threshold or within a specified range.

- [Mixture](https://preliz.readthedocs.io/en/latest/distributions/gallery/mixture.html) - A mixture distribution is a distribution that is composed of a mixture of two or more distributions. The mixture distribution is the sum of the component distributions, weighted by the probability of each component distribution.

- [Hurdle](https://preliz.readthedocs.io/en/latest/distributions/gallery/hurdle.html) - A hurdle distribution is a mixture distribution of with one process generating zeros and another non-zeros. This is different from a zero-inflated distribution, where we also have a mixture one generating zeros and another generating both zeros and positive values.

```{jupyter-execute}

dist = pz.Normal(mu=0, sigma=1)
censored_dist = pz.Censored(pz.Normal(mu=0, sigma=1), lower=1)
```

```{jupyter-execute}

dist.summary()
```

```{jupyter-execute}

censored_dist.summary()
```

```{jupyter-execute}

dist.plot_pdf()
censored_dist.plot_pdf();
```

## Integration with probabilistic programming languages

So far all the functionality we have discussed is very general and can be used in many settings where we need to work or explore with probability distributions. But the main focus of PreliZ is prior elicitation, that is how to define suitable prior distributions for Bayesian models. Thus, often PreliZ will be used together with probabilistic programming languages.

PreliZ, aims to be agnostic of probabilistic programming languages, allowing easy interaction with them if needed. Currently, there is a bias towards PyMC/Bambi, for instance, most distributions have the same parameterization as in these PPLs. But that’s more a reflection of limited dev-time than a hard design choice.

PreliZ supports exporting distributions. Currently, only PyMC and Bambi are supported.

```{jupyter-execute}

pz.Normal(0, 1).to_pymc()
```

```{jupyter-execute}

pz.Normal(0, 1).to_bambi()
```

For these methods to work you need to have installed PyMC and/or Bambi.


We can also go into the opposite direction and create PreliZ distributions from PyMC distributions, assuming we have PyMC installed.

```{jupyter-execute}
import pymc as pm

pz.from_pymc(pm.Normal.dist(mu=0, sigma=1)).summary()
```

For some functions (including `pz.plot`, `pz.maxent`, `pz.quartile`, `pz.mle`, `pz.match_moment`, `pz.match_quantiles`) you can directly pass a PyMC distribution, and it will work as expected.

```{jupyter-execute}
pz.plot(pm.Normal.dist(mu=0, sigma=1));
```

Or even

```{jupyter-execute}
with pm.Model():
    x = pm.Truncated("x", pm.NegativeBinomial.dist(2.5, 3), 0, 7)
pz.plot(x);
```

PreliZ can also work with `Prior` objects from [PyMC-extras](https://www.pymc.io/projects/extras/en/latest/generated/pymc_extras.prior.Prior.html#pymc_extras.prior.Prior), as long as we have PyMC-extras installed and the resulting distribution is implemented in PreliZ.

```{jupyter-execute}
from pymc_extras.prior import Prior

pz.plot(Prior("Gamma", mu=4, sigma=2));
```

In other examples we will discuss methods more directly focus on prior elicitation, and also other ways to interact with PPLs.

