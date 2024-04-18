# Pairs trading Kalman Filter parameter estimation utilizing the Shumway and
# Stoffer (1982) approach as described in Elliot et. al. 2005
# Andrew Pasco

import numpy as np
import pandas as pd
import statsmodels.api as sm

def kf(v, y):
    '''
    Use Kalman Filter to estimate process {x_N} mean (mu_N), cov (R_N) at time
    t_N given N-series of observed values {y_k}, v=(A,B,C**2,D**2).
    Take x^_0 = y_0, R_0 = D**2.
    Algorithm implemented as described in Elliot et. al. (2005)
    Andrew Pasco
    '''
    # Unpack params
    (a, b, c2, d2) = v
    
    # Initial Vals for mu, R
    mu =  [y[0]]
    R = [d2]

    # KF Recursion Algo
    # yields mu_N, R_N, if len(y) = N
    for i in np.arange(len(y)-1):
        # process update
        mu_22 = a + b*mu[-1] # 22
        cov_23 = R[-1]*b**2 + c2 # 23
        # gain
        k = cov_23/(cov_23 + d2) # 24
        # Posterior update
        mu.append(mu_22 + k*(y[i+1] - mu_22)) # 25
        R.append(d2*k) # 26

    return (mu, R)

def ss(mu, R, v_j, n):
    '''
    given initial values for backwards recursion mu_N, R_N (from previous KF),
    and the previous best guess for the parameters v_j, compute v_j+1 and
    the next KF initial values.
    Utilizes Shumway and Stoffer (1982) approach as described in Elliot et. al.
    '''
    # unpack params
    (a, b, c2, d2) = v_j
    mu_31 = [mu[-1]] # init with x^_N|N from KF
    cov_32 = [R[-1]] # init with R_N|N from KF
    cov_33 = []

    # this still needs some work... be careful when we are using values from kf vector and when to use backwards recursion vals
    for k in np.arange(n, 0, -1): # skip 35, 36 for the first step: we already have them!
        # for convenience
        mu_k = mu[k] # x^_k|k
        R_k = R[k] # R_k|k

        # compute smoothers
        jk = b*R_k / (R_k*b**2 + c2) # 34
        if k != n:
            mu_31.append(mu_k + jk*((a - (n-k)*b*mu_31[0]) - (a+b*mu_k))) # 35
            cov_32.append(R_k + jk**2((c2 - (n-k)*cov_32[0]*b**2) - (R_k*b**2 + c2))) # 36
            cov_33.append()
        jkm1 = jk
    pass