# Use Kalman Filter to estimate process {x_N} mean (mu_N), cov (R_N) at time
# t_N given N-series of observed values {y_k}, v=(A,B,C**2,D**2).
# Take x^_0 = y_0, R_0 = D**2.
# Algorithm implemented as described in Elliot et. al. (2005)
# Andrew Pasco

import numpy as np

def kf(v, y):
    # Unpack params
    (a, b, c2, d2) = v
    
    # Initial Vals for mu, R
    mu =  y[0]
    R = d2

    # KF Recursion Algo
    # yields mu_N, R_N, if len(y) = N
    for i in np.arange(len(y)-1):
        # process update
        mu_22 = a + b*mu
        cov_23 = R*b**2 + c2
        # gain
        k = cov_23/(cov_23 + d2)
        # Posterior update
        mu = mu_22 + k*(y[i+1] - mu_22)
        R = d2*k    