import preliz as pz
import logging

_log = logging.getLogger("preliz")


def one_iter(lower, upper, mode, mass=0.99, eps=0.005, plot=True):

    """ Fits Parameters to a Beta Distribution based on the mode, confidence intervals and mass of the distribution.

    Parameters
    -----------
    lower : float
        Lower end-point between 0 and upper

    upper : float
        Upper end-point between lower and 1

    mode : float
        Mode of the Beta distribution between lower and upper

    mass : float
        Concentarion of the probabilty mass between lower and upper. Defaults to 0.99


    eps : float
        Tolerance for the mass of the distribution. Defaults to 0.005

    plot : bool
        Whether to plot the distribution. Defaults to True.


    Returns
    --------

    dist : Preliz Beta distribution

        Beta distribution with fitted parameters alpha and beta for the given mass and intervals.

    """

    if not 0 < mass <= 1:
        raise ValueError("mass should be larger than 0 and smaller or equal to 1")

    if upper <= lower:
        raise ValueError("upper should be larger than lower")

    alpha = 1
    beta = 1
    dist = pz.Beta(alpha, beta)
    prob = dist.cdf(upper) - dist.cdf(lower)
    
    tau_not = 0
    while abs(prob - mass) > eps:

        tau_not += 0.1
        alpha = 1 + mode * tau_not
        beta = 1 + (1 - mode) * tau_not
        dist._parametrization(alpha, beta)
        prob = dist.cdf(upper) - dist.cdf(lower)

    relative_error = abs((prob - mass) / mass * 100)

    if relative_error > eps*100:
        _log.info(
            " The requested mass is %.3g, but the computed one is %.3g",
            mass,
            prob,
        )
    if plot:
        dist.plot_pdf()
    return dist



lower = 0.2
upper = 0.95
prob = 0.90
dist = one_iter(lower, upper,
                mode=0.8,
                mass=prob, 
                
                )
dist_ = pz.Beta()
pz.maxent(dist_, lower, upper, prob)
