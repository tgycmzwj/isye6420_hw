import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import pandas as pd
import arviz as az

if __name__=='__main__':
    g1=np.array([45,59,48,46,38,47])
    g2=np.array([21,12,14,17,13,17])
    g3=np.array([16,11,20,21,14,7])
    g4=np.array([37,32,15,25,39,41])
    sample_ave=np.mean(np.concatenate([g1,g2,g3,g4]))
    with pm.Model() as m:
        g1_data=pm.Data('g1_data',g1)
        g2_data=pm.Data('g2_data',g2)
        g3_data=pm.Data('g3_data',g3)
        g4_data=pm.Data('g4_data',g4)
        mu=pm.Normal('mu',mu=sample_ave,tau=0.0001)
        alpha1=pm.Normal('alpha1', mu=0, tau=0.0001)
        alpha2=pm.Normal('alpha2',mu=0,tau=0.0001)
        alpha3=pm.Normal('alpha3',mu=0,tau=0.0001)
        alpha4=pm.Deterministic('alpha4',-alpha1-alpha2-alpha3)
        alpha_diff12=pm.Deterministic('diff12',alpha1-alpha2)
        alpha_diff13=pm.Deterministic('diff13',alpha1-alpha3)
        alpha_diff14=pm.Deterministic('diff14',alpha1-alpha4)
        alpha_diff23=pm.Deterministic('diff23',alpha2-alpha3)
        alpha_diff24=pm.Deterministic('diff24',alpha2-alpha4)
        alpha_diff34=pm.Deterministic('diff34',alpha3-alpha4)
        tau=pm.Gamma('tau',alpha=0.001,beta=0.001,shape=4)
        lld1=pm.Normal('lld1',mu=mu+alpha1,tau=tau[0],observed=g1_data)
        lld2=pm.Normal('lld2',mu=mu+alpha2,tau=tau[1],observed=g2_data)
        lld3=pm.Normal('lld3',mu=mu+alpha3,tau=tau[2],observed=g3_data)
        lld4=pm.Normal('lld4',mu=mu+alpha4,tau=tau[3],observed=g4_data)
        trace = pm.sample(draws=20000, chains=4, tune=5000,
                          init="jitter+adapt_diag",
                          random_seed=4, target_accept=0.95, )
    with m:
        print(az.summary(trace,hdi_prob=0.95))
        az.summary(trace,hdi_prob=0.95).to_csv('q3.csv')
    print('finished')