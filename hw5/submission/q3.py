import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import pandas as pd

if __name__=='__main__':
    data_raw = pd.read_csv('q3_data.csv', header=None).values
    doses=[0,0.5,1,2,3,4]
    micronucleis=[0,1,2,3,4,5,6]

    #expand the dataset to two variable
    data=[]
    for i in range(len(doses)):
        for j in range(len(micronucleis)):
            dose=doses[i]
            micronuclei=micronucleis[j]
            for k in range(data_raw[i][j]):
                data.append((dose,micronuclei))
    data = pd.DataFrame(data, columns=['dose','micronuclei'])
    data['const']=1

    x=data[['const','dose']].values
    y=data['micronuclei'].values
    with pm.Model() as m:
        #associate data with model
        x_data=pm.Data('x',x)
        y_data=pm.Data('y',y)
        #priors
        b0=pm.Normal('intercept',mu=0,sigma=10e3)
        b1=pm.Normal('coef_dose',mu=0,sigma=10e3)
        micro=pm.Poisson('micro',
                         mu=np.exp(b0*x_data[:,0]+
                                   b1*x_data[:,1]),
                         observed=y_data)
        trace = pm.sample(draws=3000, chains=4,tune=500,
                          init="jitter+adapt_diag",
                          random_seed=4,target_accept=0.95, )
    print(az.summary(trace, hdi_prob=0.95))
    az.summary(trace, hdi_prob=0.95).to_csv('q3.csv')

    #prediction
    with m:
        x_pred = [1,3.5]
        new_obs = np.array([x_pred])
        pm.set_data({'x': new_obs})
        ppc = pm.sample_posterior_predictive(trace, samples=50)
        print(az.summary(ppc, hdi_prob=0.95))
        az.summary(ppc, hdi_prob=0.95).to_csv('q3_pred.csv')

    print('finished')



