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
# Truncated Distribution

This is not a distribution per se, but a modifier of univariate distributions. 

Truncated distributions arise in cases where the ability to record, or even to know about, occurrences is limited to values which lie above or below a given threshold or within a specified range. For example, if the dates of birth of children in a school are examined, these would typically be subject to truncation relative to those of all children in the area given that the school accepts only children in a given age range on a specific date. There would be no information about how many children in the locality had dates of birth before or after the school's cutoff dates if only a direct approach to the school were used to obtain information. 

## Probability Density Function (PDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Truncated Distribution PDF
---

import arviz as az
from preliz import Gamma, Truncated
az.style.use('arviz-doc')
Truncated(Gamma(mu=2, sigma=1), 1, 4.5).plot_pdf()
Gamma(mu=2, sigma=1).plot_pdf();
```

## Cumulative Distribution Function (CDF):

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Trucated Distribution CDF
---

Truncated(Gamma(mu=2, sigma=1), 1, 4.5).plot_cdf()
Gamma(mu=2, sigma=1).plot_cdf();
```


## Key properties and parameters:


**Probability Density Function (PDF):**

Given a base distribution with cumulative distribution function (CDF) and probability density mass/function (PDF). The pdf of a Truncated distribution is:

$$
\begin{cases}
    0 & \text{for } x < \text{lower}, \\
    \frac{\text{PDF}(x)}{\text{CDF}(upper) - \text{CDF}(lower)}
    & \text{for } \text{lower} <= x <= \text{upper}, \\
    0 & \text{for } x > \text{upper},
\end{cases}
$$

where `lower` and `upper` are the lower and upper bounds of the truncated distribution, respectively.

**Cumulative Distribution Function (CDF):**

The given expression can be written mathematically as:


$$
\begin{cases} 
0 & \text{if } x_i < \text{lower} \\
1 & \text{if } x_i > \text{upper} \\
\frac{\text{CDF}(x_i) - \text{CDF}(\text{lower})}{\text{CDF}(\text{upper}) - \text{CDF}(\text{lower})} & \text{if } \text{lower} \leq x_i \leq \text{upper}
\end{cases}
$$

where `lower` and `upper` are the lower and upper bounds of the truncated distribution, respectively.


```{seealso}
:class: seealso


**Related Distributions:**

- [Censored](censored_distribution.md) - In a censored distribution, values outside the range are not recorded, while in a truncated distribution, they are set to the nearest bound.
- [TruncatedNormal](truncated_normal_distribution.md) - A truncated normal distribution is a normal distribution that has been restricted to a specific range.

```

## References

- Wikipedia - [Truncated distribution](https://en.wikipedia.org/wiki/Truncated_distribution)
