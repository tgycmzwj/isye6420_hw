import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import copy

if __name__=='__main__':
    np.random.seed(123456)
    data=[(24,102.8),(32,104.5),(48,106.5),(56,107.0),(np.nan,107.1),
                     (70,105.1),(72,103.9),(75,np.nan),(80,103.2),(96,102.1)]
    y = np.array([i[1] for i in data]).reshape(len(data), -1).copy()
    y = np.nan_to_num(y, nan=-1)
    y = np.ma.masked_values(y, value=-1)
    x = np.array([i[0] for i in data]).reshape(len(data), -1).copy()
    x = np.nan_to_num(x, nan=-1)
    x = np.ma.masked_values(x, value=-1)

    mp_y=np.where(y.mask==True)[0][0]

    with pm.Model() as m:
        mu_alpha=pm.Normal("mu_alpha",0,tau=1e-6)
        tau_alpha=pm.Gamma("tau_alpha",.001,.001)
        mu_beta=pm.Normal("mu_beta",0,tau=1e-6)
        tau_beta=pm.Gamma("tau_beta",.001,.001)
        alpha=pm.Normal("alpha",mu_alpha,tau=tau_alpha,shape=x.shape[1])
        beta=pm.Normal("beta",mu_beta,tau=tau_beta,shape=x.shape[1]+1)
        tau_lld=pm.Gamma("tau_lld",.001,.001)
        x_imputed=pm.TruncatedNormal("x_imputed",mu=60,sigma=10,lower=0,observed=x)
        mu=alpha+beta[0]*x_imputed+beta[1]*x_imputed**2
        lld=pm.Normal("lld",mu,tau=tau_lld,observed=y)
        #r2
        cy = y - y.mean()
        sst = pm.math.dot(cy.squeeze(), cy.squeeze())
        sse = (10 - 3) / tau_lld
        br2 = pm.Deterministic("br2", 1 - sse / sst)
        trace=pm.sample(10000,tune=1000,cores=4,init="jitter+adapt_diag",step=[pm.NUTS(target_accept=.95)],random_seed=123456)

    with m:
        print(az.summary(trace, hdi_prob=0.95))
        az.summary(trace,hdi_prob=0.95).to_csv('q1_part2_n.csv')
        # calculate r2
        r2_collector=[]
        all_y,all_tau=np.concatenate(trace.posterior.lld_missing),np.concatenate(trace.posterior.tau_lld)
        for i in range(len(all_y)):
            imputed_y_vector=np.concatenate((y[:mp_y].squeeze(),np.array(all_y[i]),y[mp_y+1:].squeeze()))
            sst=np.sum((imputed_y_vector-np.mean(imputed_y_vector))**2)
            sse=(10-3)/all_tau[i]
            r2_collector.append(1-sse/sst)
        print('r2 is given by: '+str(np.mean(r2_collector)))
    print('part 2 finished')