import preliz as pz

def calc_cdf(l1, u1, a, b):
    dist = pz.Beta(a, b)
    return dist.cdf(u1) - dist.cdf(l1)

def one_iter(l1, u1, stop_prob=0.99, eps=0.001):

    tau_not = 0
    mode = (l1 + u1) / 2

    prob = calc_cdf(l1, u1, 1, 1)
    
    while abs(prob - stop_prob) > eps:

        tau_not +=  0.1

        alpha1 = 1 + mode * tau_not
        alpha2 = 1 + (1 - mode) * tau_not

        prob =  calc_cdf(l1, u1, alpha1, alpha2)

    return alpha1, alpha2, prob



l1 = 0.25
l2 = 0.75
mode = (l1 + l2) / 2
alpha1, alpha2, cur_prob = one_iter(l1, l2)


print("alpha1: ", alpha1)
print("alpha2: ", alpha2)
print("cur_prob: ", cur_prob)
pz.Beta(alpha1, alpha2).plot_pdf()
pz.maxent(pz.Beta(), l1, l2, 0.99);
