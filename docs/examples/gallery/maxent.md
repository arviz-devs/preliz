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
# Maximum Entropy Distributions

```{code-cell}

import matplotlib.pyplot as plt

import preliz as pz
```

```{code-cell}

pz.style.use("preliz-doc")
```

## From intervals to maximum entropy distributions

For some priors in a model, we may know or assume that most of the mass is within a certain interval. This information is useful for determining a suitable prior, but this information alone may not be enough to obtain a unique set of parameters. The following figure shows Beta distributions with 90% of the mass between 0.1 and 0.7, the dot represent the mode of the distribution. As you can see even when all these distributions satisfies that restraint they convey very different prior knowledge.

![beta_bounds](../img/beta_bounds.png)

We can add one more condition, one that is very general. We can maximize the entropy. Given two distributions the one with more entropy is the less informative one. Loosely speaking, is the most "spread" one. In the previous figure, the blue line is the one with more entropy. Having priors with maximum entropy makes sense as this guarantees that we have the less informative distribution, given a set of constraints.

In PreliZ we can compute maximum entropy priors using the function maxent. The first argument is a PreliZ distribution.

```{code-cell}

pz.maxent(pz.Beta(), lower=0.3, upper=0.8, mass=0.6)
pz.maxent(pz.Normal(), lower=0.3, upper=0.8, mass=0.6);
```

Usually, we pass uninitialized distribution to `maxent`. But we can also pass partially initialized distribution. This is useful when we want to keep one or more parameters fixed. For instance, we may want to find a Gamma distribution with a mean of 4 and with 90% of the mass between 1 and 10.

```{code-cell}

pz.maxent(pz.Gamma(mu=4), 1, 10, 0.9);
```

If you pass a distribution with all the parameters specified, like `pz.Gamma(mu=4, sigma=1)`, you will get an error saying “All parameters are fixed, at least one should be free”.

Many functions in PreliZ update distribution in place, `maxent` is no exception. So sometimes it is best to first instantiate a distribution, and they use it, like this:

```{code-cell}

dist = pz.Gamma(mu=4)
pz.maxent(dist, 1, 10, 0.9);
```

this will allow us to keep working with `dist`, for instance to get its parameters

```{code-cell}
dist.alpha, dist.beta

```

Sometimes we may want to fix some property of a distribution, that we can not fix directly by fixing a parameter. For instance, we may want to fix the mode. We can do this by passing a proper tuple to `fixed_stat`.

```{code-cell}

dist = pz.Beta()
pz.maxent(dist, 0.1, 0.7, 0.94, fixed_stat=("mode", 0.3))
dist.mode()
```

Other values that can be passed to `fixed_stat` are “mean”, “mode”, “median”, “variance”, “std”, “skewness” or “kurtosis”.

## Unsatisfiable constraints and over-restrictive constraints

It’s important to recognize that there might not be a distribution that satisfies all our constraints. If the difference between the requested and computed masses within the interval exceeds a threshold, PreliZ will issue a warning. This helps us determine whether the computed distribution is still useful or if the inputs need to be adjusted.

On the other hand, we can also have over-restrictive constraints. For instance, if we ask for a Beta distribution with 90% of the mass between 0.1 and 0.7 and a mode of 0.5. We have enough information to determine the distribution, even without maximizing the entropy. In this case, PreliZ will just return the distribution that satisfies all the constraints without complaining.

```{code-cell}

dist = pz.Normal()
pz.maxent(dist, -2, 2, 0.9, fixed_stat=("median", 2), plot=False)
```
