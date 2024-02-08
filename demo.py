import preliz as pz

def one_iter(lower, upper, mode, mass=0.99, plot=True):
    alpha = 1
    beta = 1
    dist = pz.Beta(alpha, beta)
    prob = dist.cdf(upper) - dist.cdf(lower)
    
    tau_not = 0
    while abs(prob - mass) > 0.005:

        tau_not +=  0.1
        alpha = 1 + mode * tau_not
        beta = 1 + (1 - mode) * tau_not

        dist._parametrization(alpha, beta)
        prob = dist.cdf(upper) - dist.cdf(lower)

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
pz.maxent(dist_, lower, upper, prob);
