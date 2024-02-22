from preliz.distributions import Dirichlet, Beta


def prob_approx(tau, lower_bounds, mode):

    alpha = [1 + tau * mode_i for mode_i in mode]

    a_0 = sum(alpha)
    mean_cdf = np.mean([Beta(a_i, a_0 - a_i).cdf(lbi) for a_i, mode_i, lbi in zip(alpha, mode, lower_bounds)])
    return mean_cdf, alpha


def find_tau_dir_k(mass, mode, bound):

    # We should check that the sum of mode sums to 1, otherwise we should normalize it
    # and notify the user of the new values.

    lower_bounds = np.clip(np.array(mode) - bound, 0, 1)
    target_mass = (1-mass) / 2

    # This should go in the docs
    # I print it here, just for testing
    print(f"For the marginals {mass*100}% of the mass will be "
          f"approximately around {np.array(mode)-bound} and {np.array(mode)+bound}")

    tau = 1
    new_prob, alpha = prob_approx(tau, lower_bounds, mode)

    while abs(new_prob - target_mass) > 0.0001:
        if new_prob < target_mass:
            tau -= 0.5 * tau
        else:
            tau += 0.5 * tau

        new_prob, alpha = prob_approx(tau, lower_bounds, mode)
        
    ## We need to compare the requested mode against the computed one
    ## And report the computed one if the difference is "large enough"

    return Dirichlet(alpha)


mode = [0.4, 0.3, 0.2, 0.1]
mass = 0.90
bound = 0.01

Dirichlet_dist = find_tau_dir_k(mass, mode, bound)

alpha = Dirichlet_dist.alpha
mode = (alpha-1) / (alpha.sum() - len(alpha))

print(alpha, mode, np.sum(alpha))
Dirichlet_dist.plot_pdf()
pz.Beta(alpha[0], alpha[1:].sum()).eti(mass, fmt=".3f"), pz.Beta(alpha[-1], alpha[:-1].sum()).eti(mass, fmt=".3f")
