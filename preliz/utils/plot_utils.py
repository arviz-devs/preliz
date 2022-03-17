def plot_boxlike(fitted_dist, x_vals, ref_pdf, quantiles, ax):
    """
    Plot the mean as a dot and two interquantile ranges as lines
    """
    q_s = fitted_dist.ppf(quantiles)
    mean = fitted_dist.moment(1)

    ax.plot(x_vals, ref_pdf)
    ax.plot([q_s[1], q_s[2]], [0, 0], "k", lw=4)
    ax.plot([q_s[0], q_s[3]], [0, 0], "k", lw=2)
    ax.plot(mean, 0, "w.")
