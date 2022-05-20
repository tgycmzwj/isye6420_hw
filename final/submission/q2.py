import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import arviz as az

if __name__=='__main__':
    #load data
    data=pd.read_csv('../nanowire.csv')
    x=data['x'].values
    y=data['y'].values
    #model
    with pm.Model() as m:
        #associate data with model
        x_data=pm.Data('x',x)
        y_data=pm.Data('y',y)
        #priors
        theta1=pm.LogNormal('theta1',mu=0,tau=1/10)
        theta3=pm.LogNormal('theta3',mu=0,tau=1/10)
        theta4=pm.LogNormal('theta4',mu=0,tau=1/10)
        theta2=pm.Uniform('theta2',lower=0,upper=1)
        mu_y=theta1*pm.math.exp(-theta2*x_data**2)+theta3*(1-pm.math.exp(-theta2*x_data**2))*pm.invprobit(-x_data/theta4)
        lld=pm.Poisson('lld',mu=mu_y,observed=y_data)
        trace = pm.sample(draws=100000, chains=4, tune=10000,
                          init="jitter+adapt_diag",
                          random_seed=4, target_accept=0.95, )
    with m:
        #results summary
        print(az.summary(trace,hdi_prob=0.95))
        az.summary(trace,hdi_prob=0.95).to_csv('q2_a.csv')
        #prediction
        x_pred=[2.0]
        new_obs=np.array(x_pred)
        pm.set_data({'x':new_obs})
        ppc=pm.sample_posterior_predictive(trace,samples=50)
        print(az.summary(ppc,hdi_prob=0.95))
        az.summary(ppc,hdi_prob=0.95).to_csv('q2_b.csv')
        print('finished')