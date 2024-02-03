from scipy.stats import beta
from typing import List
from scipy.optimize import bisect


def calc_cdf(r: List[float], a: float, b: float):

    cdf1 = beta.cdf(r[0], a, b)
    cdf2 = beta.cdf(r[1], a, b)

    return cdf2-cdf1

def tau_func(tau_not):

    l1 = 0.25
    u1 = 0.75

    stop_prob = 0.99

    mode = (l1 + u1) / 2

    alpha1 = 1 + mode * tau_not
    alpha2 = 1 + (1 - mode) * tau_not

    prob = calc_cdf([l1, u1], alpha1, alpha2)

    return prob-stop_prob


def calc_beta_from_tau(tau_not, l1, u1):

        mode = (l1 + u1) / 2

        alpha1 = 1 + mode * tau_not
        alpha2 = 1 + (1 - mode) * tau_not

        prob = calc_cdf([l1, u1], alpha1, alpha2)

        return prob


def one_iter(tau_not, l1, u1, stop_prob):


    mode = (l1 + u1) / 2

    alpha1 = 1 + mode * tau_not
    alpha2 = 1 + (1 - mode) * tau_not

    prob = calc_cdf([l1, u1], alpha1, alpha2)

    while prob < stop_prob:

        tau_not = tau_not + 4

        alpha1 = 1 + mode * tau_not
        alpha2 = 1 + (1 - mode) * tau_not

        prob = calc_cdf([l1, u1], alpha1, alpha2)

    return tau_not



if __name__ == '__main__':

    l1 = 0.25
    l2 = 0.75

    eps = 0.005

    stop_prob = 0.99

    mode = 0.5

    tau_not = 0

    cur_prob = calc_beta_from_tau(tau_not,l1,l2)

    tau_star = one_iter(tau_not, l1, l2, 0.99)

    tau_bisected = bisect(tau_func, tau_not, tau_star)

    aplha1 = 1 + mode * tau_bisected
    alpha2 = 1 + (1 - mode) * tau_bisected

    cur_prob = calc_cdf([l1, l2], aplha1, alpha2)

    print("tau_star: ", tau_star)
    print("tau_bisected: ", tau_bisected)
    print("alpha1: ", aplha1)
    print("alpha2: ", alpha2)
    print("cur_prob: ", cur_prob)

