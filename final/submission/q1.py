import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import pandas as pd
import arviz as az

if __name__=='__main__':
    #load data
    data=pd.read_csv('../ortho.csv')
    data['const']=1
    #a lookup table for each unique child
    children=data['Subject'].unique()
    children_lookup=dict(zip(children,range(len(data['Subject']))))
    #create local copies of variables
    child=data['Subject'].replace(children_lookup).values
    y=data['y'].values
    x=data[['const','age','Sex_coded']].values
    #coords
    coords={'obs_id':np.arange(data.shape[0]),'child':children}
    #model with random effect
    with pm.Model(coords=coords) as m1:
        child_index1=pm.Data('child_index',child,dims='obs_id')
        x_data1=pm.Data('x_data1',x)
        y_data1=pm.Data('y_data1',y)
        beta1=pm.Normal('beta1',mu=0,sigma=10000,shape=x.shape[1])
        tau_e1=pm.Gamma('tau_e1',alpha=0.01,beta=0.01)
        tau_u1=pm.Gamma('tau_u1',alpha=0.01,beta=0.01)
        u1=pm.Normal('u1',mu=0,tau=tau_u1,dims='child')
        mu_y1=u1[child_index1]+beta1[0]*x_data1[:,0]+beta1[1]*x_data1[:,1]+beta1[2]*x_data1[:,2]
        lld1=pm.Normal('lld1',mu=mu_y1,tau=tau_e1,observed=y_data1,dims='obs_id')
        rho1=pm.Deterministic('rho1',(1/tau_u1)/(1/tau_u1+1/tau_e1))
        sigmae21=pm.Deterministic('sigmae2',1/tau_e1)
        trace1 = pm.sample(draws=100000, chains=4, tune=10000, init="jitter+adapt_diag", random_seed=4, target_accept=0.95)
    #plot results
    with m1:
        print(az.summary(trace1,hdi_prob=0.95))
        az.summary(trace1,hdi_prob=0.95).to_csv('q1_part1.csv')
        #density of rho
        v1,v2,v3,v4,v5,v6=np.concatenate(trace1.posterior['rho1']),\
                       np.concatenate(trace1.posterior['beta1'][:,:,0]),\
                       np.concatenate(trace1.posterior['beta1'][:,:,1]),\
                       np.concatenate(trace1.posterior['beta1'][:,:,2]),\
                       1/np.concatenate(trace1.posterior['tau_e1']),\
                       1/np.concatenate(trace1.posterior['tau_u1'])
        results_df=pd.DataFrame([v1,v2,v3,v4,v5])
        results_df.to_csv('q1_a_values.csv')
        plt.hist(v1,bins=40)
        plt.savefig('q1_b_rho.pdf')
        plt.show()
        plt.hist(v2,bins=40)
        plt.savefig('q1_a_beta0.pdf')
        plt.show()
        plt.hist(v3,bins=40)
        plt.savefig('q1_a_beta1.pdf')
        plt.show()
        plt.hist(v4,bins=40)
        plt.savefig('q1_a_beta2.pdf')
        plt.show()
        plt.hist(v5,bins=40)
        plt.savefig('q1_a_sigmae.pdf')
        plt.show()
        plt.hist(v6,bins=40)
        plt.savefig('q1_a_sigmau.pdf')
        plt.show()
    #model without random effect
    with pm.Model(coords=coords) as m2:
        x_data2=pm.Data('x_data2',x)
        y_data2=pm.Data('y_data2',y)
        beta2=pm.Normal('beta2',mu=0,sigma=10000,shape=x.shape[1])
        tau_e2=pm.Gamma('tau_e2',alpha=0.01,beta=0.01)
        tau_u2=pm.Gamma('tau_u2',alpha=0.01,beta=0.01)
        mu_y2=beta2[0]*x_data2[:,0]+beta2[1]*x_data2[:,1]+beta2[2]*x_data2[:,2]
        lld2=pm.Normal('lld2',mu=mu_y2,tau=tau_e2,observed=y_data2)
        sigmae22 = pm.Deterministic('sigmae2', 1 / tau_e2)
        trace2 = pm.sample(draws=100000, chains=4, tune=10000, init="jitter+adapt_diag", random_seed=456, target_accept=0.95)
    #plot results
    with m2:
        print(az.summary(trace2,hdi_prob=0.95))
        az.summary(trace2,hdi_prob=0.95).to_csv('q1_part2.csv')
        v6,v7,v8,v9=np.concatenate(trace2.posterior['beta2'][:,:,0]),\
                       np.concatenate(trace2.posterior['beta2'][:,:,1]),\
                       np.concatenate(trace2.posterior['beta2'][:,:,2]),\
                       1/np.concatenate(trace2.posterior['tau_e2'])
        results_df=pd.DataFrame([v6,v7,v8,v9])
        results_df.to_csv('q1_b_values.csv')

        plt.hist(v6,bins=40)
        plt.savefig('q1_c_beta0.pdf')
        plt.show()
        plt.hist(v7,bins=40)
        plt.savefig('q1_c_beta1.pdf')
        plt.show()
        plt.hist(v8,bins=40)
        plt.savefig('q1_c_beta2.pdf')
        plt.show()
        plt.hist(v9,bins=40)
        plt.savefig('q1_c_sigmae.pdf')
        plt.show()
    print('finished')