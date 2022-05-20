import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import statsmodels.api as sm
import random
random.seed(12345)

if __name__=='__main__':
    data=pd.read_csv('q2_data.csv',header=None)
    header=['location','gender','m1','m2',
            'm3','m4','m5','m6','m7','m8','m9']
    data.set_axis(header, axis=1, inplace=True)
    data['const']=1

    #only keep the following variables
    #y(location), x(gender,x3,x7)
    y=data['location']
    x=data[['gender','m3','m7']].values
    xc=data[['const','gender','m3','m7']].values

    #sample for prediction
    x_pred=[1,5.28,1.78]
    xc_pred=[1,1,5.28,1.78]

    #frequentist results---with constant term
    f_model1=sm.Logit(y,xc).fit()
    print(f_model1.summary())
    print(f_model1.predict(xc_pred))
    # bayesian regression---with constant term
    with pm.Model() as m:
        xc_data=pm.Data('x',xc)
        y_data=pm.Data('y',y)
        #priors
        beta=pm.Normal('beta',mu=0,sigma=10,shape=xc.shape[1])
        #likelihood
        likelihood=pm.invlogit(beta[0]*xc_data[:,0]+
                               beta[1]*xc_data[:,1]
                               +beta[2]*xc_data[:,2]+
                               beta[3]*xc_data[:,3])
        location=pm.Bernoulli(name='location',p=likelihood,observed=y_data)
        trace=pm.sample(draws=20000,chains=4,tune=1000,
                        init="jitter+adapt_diag",
                        random_seed=4,target_accept=0.95,)
    print(az.summary(trace,hdi_prob=0.95))
    az.summary(trace, hdi_prob=0.95).to_csv('q2_reg.csv')

    #prediction
    new_obs=np.array([xc_pred])
    pm.set_data({'x':new_obs},model=m)
    ppc=pm.sample_posterior_predictive(trace,model=m,samples=50)
    print(az.summary(ppc,hdi_prob=0.95))
    #manual prediction with likelihood
    select_index=np.random.randint(0,trace['beta'].shape[0],500)
    resulted_likelihood=[]
    def cal_like(coeff,input_v):
        p=np.dot(coeff,input_v)
        return np.exp(p)/(1+np.exp(p))
    for i in range(len(select_index)):
        resulted_likelihood.append(cal_like(trace['beta'][i], xc_pred))
    print('mean, 2.5, 97.5 are '+
          str(np.average(resulted_likelihood))+
          ', '+str(np.percentile(resulted_likelihood,2.5))+
          ', '+str(np.percentile(resulted_likelihood,97.5))+
          'respectively')


    print('finished')