from preliz.distributions import Dirichlet
import numpy as np
import  numpy.random as npr

#set seeds for reproducibility
npr.seed(0)


def prob_approx(num_monte_carlo_samples, tau, lower_bounds):

        # 1. Generate num_monte_carlo_samples samples from the Dirichlet distribution with parameters tau * lower_bounds
        # 2. Compute the probability that the sample is in the lower_bounds interval
        # 3. Return the probability

        k = len(lower_bounds)
        alphas = []
        sum_lower_bounds = sum(lower_bounds)
        for i in range(k):
            x_i = lower_bounds[i] + (1 - sum_lower_bounds) / k
            alphas.append(1 + tau * x_i)

        alphas = np.array(alphas)
        samples = npr.dirichlet(alphas, num_monte_carlo_samples)
        #find the number of samples that satifisfy allthe lower bounds across all dimensions
        num_satisfy = 0
        for i in range(num_monte_carlo_samples):
            if all([samples[i][j] > lower_bounds[j] for j in range(k)]):
                num_satisfy += 1
        return num_satisfy / num_monte_carlo_samples

def find_tau_bound(num_monte_carlo_samples, gamma, lower_bounds):

    tau = 1

    iter_count = 0

    while prob_approx(num_monte_carlo_samples, tau, lower_bounds) < gamma:
        tau = tau * 2
        iter_count += 1

    return tau/2, tau, iter_count

def find_tau_dir_k(gamma, lower_bounds, max_iter, num_monte_carlo_samples):

    k = len(lower_bounds)

    tau_lower, tau_upper, iter_count = find_tau_bound(num_monte_carlo_samples, gamma, lower_bounds)

    tau = (tau_lower + tau_upper) / 2

    new_prob = prob_approx(num_monte_carlo_samples, tau, lower_bounds)

    tau_value_list = [tau]
    prob_value_list = [new_prob]

    while abs(new_prob-gamma) > 0.005:
        iter_count += 1

        if new_prob > gamma:
            tau_upper = tau

        else:
            tau_lower = tau

        if tau_upper == tau_lower:
            tau_upper = tau_upper * 2

        tau = (tau_lower + tau_upper) / 2

        new_prob = prob_approx(num_monte_carlo_samples, tau, lower_bounds)

        tau_value_list.append(tau)
        prob_value_list.append(new_prob)

        if iter_count > max_iter:
            break

    alphas_list = []
    x_i_list = []

    for i in range(k):
        x_i = lower_bounds[i] + (1 - sum(lower_bounds)) / k
        x_i_list.append(x_i)
        alphas_list.append(1 + tau * x_i)

    alphas_list = np.array(alphas_list)

    Dirichlet_dist = Dirichlet(alphas_list)

    return Dirichlet_dist, tau, x_i_list, alphas_list

#write driver code to test the function

lower_bounds = [0.2, 0.2, 0.3, 0.2]
gamma = 0.99
max_iter = 1000
num_monte_carlo_samples = 1000

Dirichlet_dist, tau, x_i_list, alphas_list = find_tau_dir_k(gamma, lower_bounds, max_iter, num_monte_carlo_samples)

print("Dirichlet distribution with alpha values: ", alphas_list)
print("Tau value: ", tau)





