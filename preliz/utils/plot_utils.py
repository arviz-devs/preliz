def plot_boxlike(fitted_dist, x_vals, ref_pdf, quantiles, ax):
    """
    Plot the mean as a dot and two interquantile ranges as lines
    """
    qs = fitted_dist.ppf(quantiles)
    mean = fitted_dist.moment(1)
    support = fitted_dist.support()

    ax.plot(x_vals, ref_pdf)
    ax.plot([qs[1], qs[2]], [0, 0], "k", lw=4)
    ax.plot([qs[0], qs[3]], [0, 0], "k", lw=2)
    ax.plot(mean, 0, "w.")
