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
        mu_22 = a + b*mu[-1] # (22)
        cov_23 = R[-1]*b**2 + c2 # (23)
        # gain
        k = cov_23/(cov_23 + d2) # (24)
        # Posterior update
        mu.append(mu_22 + k*(y[i+1] - mu_22)) # (25)
        R.append(d2*k) # (26)

    return (mu, R)

def ss(mu, R, v_j, n, y):
    '''
    given initial values for backwards recursion mu_N, R_N (from previous KF),
    and the previous best guess for the parameters v_j, compute v_j+1 and
    the next KF initial values.
    Utilizes Shumway and Stoffer (1982) approach as described in Elliot et. al.
    '''
    # unpack params
    (a, b, c2, d2) = v_j
    mu_31 = [mu[-1]] # init with x^_N|N from KF (31) NB: REVERSE INDEXED
    cov_32 = [R[-1]] # init with R_N|N from KF (32) NB: REVERSE INDEXED
    cov_33 = [] # gets init with Sig_N-1,N|N in first loop (33) NB: REVERSE INDEXED

    # compute smoother vectors
    for k in np.arange(n, 0, -1): # skip 35, 36 for the first step: we already have them!
        # for convenience
        mu_k = mu[k] # x^_k|k
        R_k = R[k] # R_k|k

        # compute smoothers
        jk = b*R[k] / (R[k]*b**2 + c2) # (34)
        jkm1 = b*R[k-1] / (R[k-1]*b**2 + c2)
        if k != n:
            mu_31.append(mu[k] + jk*(mu_31[-1] - (a+b*mu[k]))) # (35)
            cov_32.append(R[k] + jk**2(cov_32[-1] - (R[k]*b**2 + c2))) # (36)
            cov_33.append(jkm1*R[k] + jk*jkm1*()) # (37)
        else:
            cov_33.append(b*(1-(R[-1]/d2))*R[-2]) # (38), taking K_N as R_N/d2 as in (26)
    
    mu_31 = np.array(mu_31[::-1]) # fix order of indexing
    cov_32 = np.array(cov_32[::-1]) # fix order of indexing
    cov_33 = np.array(cov_33[::-1]) # fix order of indexing
    y = np.array(y)

    # Compute parameter update
    alph = np.sum(cov_32[:-1] + mu_31[:-1]**2)
    bet = np.sum(cov_33[1:] + mu_31[:-1]*mu[1:])
    gam = np.sum(mu[1:])
    delt = gam - mu_31[-1] + mu_31[0]

    a_new = (alph*gam - delt*bet)/(n*alph - delt**2) # (39)
    b_new = (n*bet - gam*delt)/(n*alph - delt**2) # (40)
    c2_new = 1/n * np.sum(cov_32[1:] + mu_31[1:]**2 + a_new**2 +       # (41)
                          cov_32[:-1]*b_new**2 + (b_new*mu_31[:-1])**2 -
                          2*a_new*mu_31[1:] + 2*a_new*b_new*mu_31[:-1] - 
                          2*b_new*cov_33[:-1] - 2*b_new*mu_31[1:]*mu_31[:-1])
    d2_new = 1/(n+1) * np.sum(y**2 - 2*y*mu_31 + cov_32 + mu_31**2) # (42)
    
    return (a_new, b_new, c2_new, d2_new)


def gen_obs(v, n):
    # generate n noisy observations {yk} of the process {xk}
    # process is as shown in (18), (19)
    (a, b, c2, d2) = v
    
    x = [1] # arbitrary process init
    y = [x[0] + np.random.randn()] # noisy observation init

    for _ in np.arange(n-1): # generate process
        x.append(a + b*x[-1] + np.sqrt(c2)*np.random.randn())
        y.append(x[-1] + np.random.randn()) # noisy observations

    return y


