import matplotlib.pyplot as plt
import numpy as np

from preliz import Dirichlet, MvNormal, style

style.use("preliz-doc")

w = 1834 / 300
h = 1234 / 300

mu = [0.0, 0]
sigma = np.array([[1, 1], [1, 1.5]])
_, ax = plt.subplots()
ax = MvNormal(mu, sigma).plot_pdf(marginals=False, legend=False, ax=ax)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[["left", "bottom"]].set_visible(False)
plt.savefig("distributions/img/MvNormal.png")

_, ax = plt.subplots(figsize=(w, h))
ax = Dirichlet([5, 5, 5]).plot_pdf(marginals=False, legend=None, ax=ax)
ax.get_lines()[0].set_alpha(0)
ax.set_xticks([])
ax.set_yticks([])
ax.spines[["left", "bottom"]].set_visible(False)
plt.savefig("distributions/img/Dirichlet.png")
