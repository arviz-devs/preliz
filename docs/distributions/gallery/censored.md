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
(Censored_gallery)=
# Censored Distribution

<audio controls> <source src="../../_static/censored.mp3" type="audio/mpeg"> This browser cannot play the pronunciation audio file for this distribution. </audio>

[Modifier](../../gallery_tags.rst#modifier)

This is not a distribution per se, but a modifier of univariate distributions.

A censored distribution arises when the observed data is limited to a certain range, and values outside this range are not recorded. For instance, in a study aiming to measure the impact of a drug on mortality rates it may be known that an individual's age at death is at least 75 years (but may be more). Such a situation could occur if the individual withdrew from the study at age 75, or if the individual is currently alive at the age of 75. Censoring can also happen when a value falls outside the range of a measuring instrument. For example, if a bathroom scale only measures up to 140 kg, and a 160-kg person is weighed, the observer would only know that the individual's weight is at least 140 kg.

## Key properties and parameters

**Parameters:**

- `dist` : (PreliZ distribution) Univariate distribution to be censored.
- `lower` : (float, int, or `np.inf`) Lower (left) censoring point, `np.inf` indicates no lower censoring.
- `upper` : (float, int, or `np.inf`) Upper (right) censoring point, `np.inf` indicates no upper censoring.

### Probability Density Function (PDF)

Given a base distribution with cumulative distribution function (CDF) and probability density mass/function (PDF). The pdf of a Censored distribution is:

$$
\begin{cases}
    0 & \text{for } x < \text{lower}, \\
    \text{CDF}(\text{lower}) & \text{for } x = \text{lower}, \\
    \text{PDF}(x) & \text{for } \text{lower} < x < \text{upper}, \\
    1-\text{CDF}(\text{upper}) & \text {for } x = \text{upper}, \\
    0 & \text{for } x > \text{upper},
\end{cases}
$$

where `lower` and `upper` are the lower and upper bounds of the censored distribution, respectively.

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Censored Distribution PDF
---


from preliz import Normal, Censored, style
style.use('preliz-doc')
Censored(Normal(0, 1), -1, 1).plot_pdf(support=(-4, 4))
Normal(0, 1).plot_pdf(alpha=0.5);
```

### Cumulative Distribution Function (CDF)

The given expression can be written mathematically as:

$$
\begin{cases}
    0 & \text{for } x < \text{lower}, \\
    \text{CDF}(x) & \text{for } \text{lower} < x < \text{upper}, \\
    1 & \text{for } x > \text{upper},
\end{cases}
$$

where `lower` and `upper` are the lower and upper bounds of the censored distribution, respectively.

```{code-cell}
---
tags: [remove-input]
mystnb:
  image:
    alt: Censored Distribution CDF
---

Censored(Normal(0, 1), -1, 1).plot_cdf(support=(-4, 4))
Normal(0, 1).plot_cdf(alpha=0.5);
```

```{seealso}
:class: seealso


**Related Distributions:**

- [Truncated](truncated.md) - In a truncated distribution, values outside the range are set to the nearest bound, while in a censored distribution, they are not recorded.

```

## References

- Wikipedia - [Censored distribution](https://en.wikipedia.org/wiki/Censoring_(statistics))