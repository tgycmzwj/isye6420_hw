import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import copy
from pymc.math import exp


if __name__=='__main__':
    data=pd.read_csv('bladderc.csv')
    y=np.array(data['time'])
    x=np.array(data['group'])
    censored_vals=np.array(data['observed'])
    #separate the observed values and the censored values
    observed_mask=censored_vals==1
    censored=y[~observed_mask]
    y_uncensored=y[observed_mask]
    x_censored = x[~observed_mask]
    x_uncensored = x[observed_mask]
    #model
    with pm.Model() as m:
        beta0 = pm.Normal("beta0", 1, tau=0.0001)
        beta1 = pm.Normal("beta1", 0, tau=0.0001)
        lambda_censored = exp(beta0 + beta1 * x_censored)
        lambda_uncensored = exp(beta0 + beta1 * x_uncensored)
        impute_censored = pm.Bound("impute_censored",
                         pm.Exponential.dist(lam=lambda_censored),lower=censored,shape=censored.shape[0],
                     )
        likelihood = pm.Exponential("likelihood",
                        lam=lambda_uncensored,observed=y_uncensored,shape=y_uncensored.shape[0],
                    )
        median0 = pm.Deterministic("mu0", exp(-beta0))
        median1 = pm.Deterministic("mu1", exp(-beta0 - beta1))
        mudiff=pm.Deterministic('mudiff',median1-median0)
        effectice=pm.Deterministic('effective',median0<median1)
        trace = pm.sample(10000, tune=1000, cores=4, init="auto", step=[pm.NUTS(target_accept=0.95)])
    with m:
        print(az.summary(trace, hdi_prob=0.9))
        az.summary(trace, hdi_prob=0.9).to_csv('q2.csv')
    #probability that it works

    print('finished')

