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

# Direct Elicitation in ND

```{jupyter-execute}

import preliz as pz
```

```{jupyter-execute}

pz.style.use("preliz-doc")
```

## Dirichlet mode

`dirichlet_mode` returns a Dirichlet distribution where the marginals have the specified mode and mass and their masses
lie within the range mode ± bound.

The mode of a Dirichlet is defined as:

$$x_i = \frac{\alpha_i - 1}{\alpha_0 - K}, \qquad \alpha_i > 1$$

where $K$ is the number of categories (the length of the vector $\alpha$), and $\alpha_0 = \sum_{i=1}^K\alpha_i$

```{jupyter-execute}

ax, dist = pz.dirichlet_mode([1/3, 1/3, 1/3], bound=0.2)
```

When the dimension of the Dirichlet is 3 we can also plot the joint pdf

```{jupyter-execute}

dist.plot_pdf(marginals=False);
```

```{jupyter-execute}

ax, dist = pz.dirichlet_mode([0.2, 0.3, 0.5], bound=0.1)
```

```{jupyter-execute}

dist.plot_pdf(marginals=False);
```
