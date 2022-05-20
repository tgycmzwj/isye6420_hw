import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('seaborn')
np.random.seed(42)

if __name__=='__main__':
    #read_data
    data=pd.read_csv('birth.csv')
    data=data[['t','num']]
    #plot graph for the data
    plt.plot(data['t'],data['num'])
    plt.xlabel('Time (weeks)')
    plt.ylabel('Number of Birth (1,000)')
    plt.savefig('graph1.pdf')
    plt.show()
    #split training and testing sample
    num_train=300
    data_train,data_test=data[:num_train],data[num_train:]
    #use frequelist regression
    data_train['time']='2000-01-01'
    data_train['time']=pd.to_datetime(data_train['time'])
    data_train['time']=data_train['time']+pd.to_timedelta(data_train['t']*7,unit='days')
    data_train_freq=data_train.set_index('time')
    data_train_freq=data_train_freq[['num']]
    # decomp = seasonal_decompose(data_train_freq["num"])
    # decomp.plot()
    # plt.savefig('decompose.pdf')
    # plt.show()
    stl = STL(data_train_freq, seasonal=13)
    res=stl.fit()
    fig,ax=plt.subplots(3)
    ax[0].plot(res.observed)
    ax[0].set_ylabel('observed')
    ax[1].plot(res.trend)
    ax[1].set_ylabel('trend')
    ax[2].plot(res.seasonal)
    ax[2].set_ylabel('seasonal')
    # fig=res.plot()
    plt.savefig('decompose.pdf')
    plt.show()
    stlf = STLForecast(data_train_freq, ARIMA, model_kwargs=dict(order=(1, 1, 0)))
    stlf_res = stlf.fit()
    forecast = stlf_res.forecast(100)
    forecast=pd.DataFrame(forecast)
    forecast.reset_index(inplace=True)
    forecast.rename(columns={0:'num'},inplace=True)
    forecast=forecast[['num']]
    data_test_freq=data_test.reset_index()
    data_test_freq=data_test_freq[['num']]
    #training r2
    train_pred=res.seasonal+res.trend
    r2_train=1-np.sum((train_pred-data_train_freq['num'])**2)/np.sum((data_train['num']-data_train['num'].mean())**2)
    #testing r2
    r2_test=1-np.sum((forecast['num']-data_test_freq['num'])**2)/np.sum((data_test['num']-data_test['num'].mean())**2)
    train_pred=pd.DataFrame(train_pred)
    train_pred.reset_index(inplace=True)
    train_pred.rename(columns={0:'num'},inplace=True)
    train_pred=train_pred[['num']]
    #plot: training forecast
    plt.plot(train_pred,label='prediction')
    plt.plot(data_train['num'],label='True Value')
    plt.legend()
    plt.savefig('prediction_training.pdf')
    plt.show()
    #plot: testing forecast
    plt.plot(forecast['num'],label='prediction')
    plt.plot(data_test_freq['num'],label='True Value')
    plt.legend()
    plt.savefig('prediction_testing.pdf')
    plt.show()
    print('training r2 of frequenlist model is given by '+str(r2_train))
    print('testing r2 of frequenlist model is given by '+str(r2_test))







    #bayesian model
    with pm.Model() as m:
        #gaussian process: two seasonality term with one linear trend
        sns1=pm.Gamma(name='sns1',alpha=2,beta=2)
        cycle1=pm.Gamma(name='cycle1',alpha=52,beta=1)
        gauss1=pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1,ls=sns1,period=cycle1))
        sns2=pm.Gamma(name='sns2',alpha=1,beta=1)
        cycle2=pm.Gamma(name='cycle2',alpha=15,beta=1)
        gauss2=pm.gp.Marginal(cov_func=pm.gp.cov.Periodic(input_dim=1,ls=sns2,period=cycle2))
        linear=pm.Normal(name='linear',mu=0.1,sigma=1)
        gauss3=pm.gp.Marginal(cov_func=pm.gp.cov.Linear(input_dim=1,c=linear))
        gauss=gauss1+gauss2+gauss3
        #observation
        sigma=pm.Gamma(name='sigma',alpha=0.1,beta=0.1)
        y_prediction = gauss.marginal_likelihood('y_prediction', X=data_train['t'].values.reshape(num_train,1), y=data_train['num'].values.reshape(num_train,1).flatten(), noise=sigma)
        trace = pm.sample(draws=2000, chains=4, tune=500)
    with m:
        #print results
        print(az.summary(trace,hdi_prob=0.95))
        az.summary(trace,hdi_prob=0.95).to_csv('bayesian_summary.csv')
        #plot results
        az.plot_trace(trace)
        plt.savefig('bayesian_trace.pdf')
        plt.show()
        #generate prediction
        x_train_values = gauss.conditional('x_train_values', data_train['t'].values.reshape(num_train,1))
        y_train_pred = pm.sample_posterior_predictive(trace, var_names=['x_train_values'], samples=500)
        x_test_values = gauss.conditional('x_test_values', data_test['t'].values.reshape(data_test.shape[0],1))
        y_test_pred = pm.sample_posterior_predictive(trace, var_names=['x_test_values'], samples=500)
    #plot predictions
    y_train_pp,y_test_pp=y_train_pred['x_train_values'],y_test_pred['x_test_values']
    y_train_mean,y_train_025,y_train_975,y_test_mean,y_test_025,y_test_975=y_train_pp.mean(axis=0),np.percentile(y_train_pp,0.25,axis=0),np.percentile(y_train_pp,97.5,axis=0),y_test_pp.mean(axis=0),np.percentile(y_test_pp,0.25,axis=0),np.percentile(y_test_pp,97.5,axis=0)
    #plot: training sample
    fig, ax = plt.subplots()
    ax.plot(data_train['t'], y_train_mean, color='steelblue',label='Prediction')
    ax.fill_between(data_train['t'], y_train_025, y_train_975, color='lightslategray', alpha=.2, label='confidence')
    ax.plot(data_train['t'],data_train['num'],color='green',alpha=0.5,label='True Value')
    ax.legend()
    plt.savefig('bayesian_prediction_training.pdf')
    plt.show()
    #plot: testing sample
    fig, ax = plt.subplots()
    ax.plot(data_test['t'], y_test_mean, color='steelblue',label='Prediction')
    ax.fill_between(data_test['t'], y_test_025, y_test_975, color='lightslategray', alpha=.2, label='confidence')
    ax.plot(data_test['t'],data_test['num'],color='green',alpha=0.5,label='True Value')
    ax.legend()
    plt.savefig('bayesian_prediction_testing.pdf')
    plt.show()
    #calculate training r2
    br2_training=1-np.sum((y_train_mean-data_train['num'])**2)/np.sum((data_train['num']-data_train['num'].mean())**2)
    #calculate testing r2
    br2_testing=1-np.sum((y_test_mean-data_test['num'])**2)/np.sum((data_test['num']-data_test['num'].mean())**2)
    print('training r2 of bayesian model is given by '+str(br2_training))
    print('testing r2 of bayesian model is given by '+str(br2_testing))


    print('finished!')
