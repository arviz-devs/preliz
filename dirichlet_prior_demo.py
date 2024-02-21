from preliz.distributions import Dirichlet
import numpy as np


def prob_approx(tau, lower_bounds, avg_rem_mass):

    k = len(lower_bounds)
    alpha = [1 + tau * (lower_bounds[i] + avg_rem_mass) for i in range(k)]

    return np.mean(Dirichlet(alpha).cdf(lower_bounds)), alpha

def find_tau_bound(mass, lower_bounds, avg_rem_mass):
    tau = 1
    iter_count = 0
    while prob_approx(tau, lower_bounds, avg_rem_mass)[0] < mass:
        tau *= 2
        iter_count += 1

    return tau / 2, tau, iter_count


def find_tau_dir_k(mass, mode):

    # we should check that the sum of mode sum to 1, otherwise we should normalize it 
    # and notify the user the new values.

    avg_rem_mass = (1- mass) / len(mode)
    lower_bounds = np.clip(np.array(mode) - avg_rem_mass, 0, 1)

    tau_lower, tau_upper, iter_count = find_tau_bound(mass, lower_bounds, avg_rem_mass)
    tau = (tau_lower + tau_upper) / 2
    new_prob, alpha = prob_approx(tau, lower_bounds, avg_rem_mass)

    while abs(new_prob - mass) > 0.0005:
        iter_count += 1

        if new_prob > mass:
            tau_upper = tau
        else:
            tau_lower = tau
        
        if tau_upper == tau_lower:
            tau_upper = tau_upper * 2

        tau = (tau_lower + tau_upper) / 2

        new_prob, alpha = prob_approx(tau, lower_bounds, avg_rem_mass)

    return Dirichlet(alpha)


mode = [0.4, 0.2, 0.2, 0.2]
mass = 0.90

Dirichlet_dist = find_tau_dir_k(mass, mode)

alpha = Dirichlet_dist.alpha
mode = (alpha-1) / (alpha.sum() - len(alpha))

alpha, mode
