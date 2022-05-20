import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az


#data
if __name__=='__main__':
    #sample
    data_clamp=[13.8,8.0,8.4,8.8,9.6,9.8,8.2,
                8.0,10.3,8.5,11.5,8.2,8.9,9.4,10.3,12.6]
    data_unclamp=[10.4,13.1,11.4,9.0,11.9,16.2,
                  14.0,8.2,13.0,8.8,14.9,12.2,11.2,13.9,13.4,11.9]
    np.average(data_clamp)-np.average(data_unclamp)
    #prior parameters
    shape=0.001
    rate=0.001
    with pm.Model() as m:
        #priors
        alpha1=pm.Gamma('alpha1',alpha=shape,beta=rate)
        beta1=pm.Gamma('beta1',alpha=shape,beta=rate)
        alpha2 = pm.Gamma('alpha2', alpha=shape, beta=rate)
        beta2 = pm.Gamma('beta2', alpha=shape, beta=rate)
        mudiff = pm.Deterministic('mudiff',alpha1/beta1-alpha2/beta2)
        #likelihood
        y1=pm.Gamma('y1',alpha=alpha1,beta=beta1,observed=data_clamp)
        y2=pm.Gamma('y2',alpha=alpha2,beta=beta2,observed=data_unclamp)
        #start sampling
        trace=pm.sample(20000,chains=4,tune=1000,
                        init="jitter+adapt_diag",
                        random_seed=1,return_inferencedata=True,)
        #print results
        print(az.summary(trace,hdi_prob=0.95))
        az.summary(trace, hdi_prob=0.95).to_csv('q1.csv')

















