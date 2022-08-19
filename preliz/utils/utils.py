import numpy as np
from scipy.special import gamma


def hdi_from_pdf(dist, mass=0.95):
    """
    Approximate the HDI by evaluating the pdf.
    This is faster, but potentially less accurate, than directly minimizing the
    interval as evaluating the ppf can be slow, specially for some distributions.
    """
    if dist.kind == "continuous":
        lower_ep, upper_ep = dist._finite_endpoints("full")
        x_vals = np.linspace(lower_ep, upper_ep, 10000)
        pdf = dist.rv_frozen.pdf(x_vals)
        pdf = pdf / pdf.sum()
    else:
        x_vals = dist.xvals(support="full")
        pdf = dist.rv_frozen.pmf(x_vals)

    sorted_idx = np.argsort(pdf)[::-1]
    mass_cum = 0
    indices = []
    for idx in sorted_idx:
        mass_cum += pdf[idx]
        indices.append(idx)
        if mass_cum >= mass:
            break
    return x_vals[np.sort(indices)[[0, -1]]]


def garcia_approximation(mean, sigma):
    """
    Approximate method of moments for Weibull distribution, provides good results for values of
    alpha larger than 0.83.

    Oscar Garcia. Simplified method-of-moments estimation for the Weibull distribution. 1981.
    New Zealand Journal of Forestry Science 11:304-306
    https://www.scionresearch.com/__data/assets/pdf_file/0010/36874/NZJFS1131981GARCIA304-306.pdf
    """
    ks = [-0.221016417, 0.010060668, 0.117358987, -0.050999126]  # pylint: disable=invalid-name
    z = sigma / mean  # pylint: disable=invalid-name

    poly = 0
    for idx, k in enumerate(ks):
        poly += k * z**idx

    alpha = 1 / (z * (1 + (1 - z) ** 2 * poly))
    beta = 1 / (gamma(1 + 1 / alpha) / (mean))
    return alpha, beta
