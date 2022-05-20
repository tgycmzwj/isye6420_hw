import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
#statistics
obs=[0,2,2,3,3,1,2,1,1]
n_sample=100000
n_burn=1000
a=2
lam=1

#posterior density function
a_delta=sum(obs)
b_delta=len(obs)


#first case: we assume the distribution of b is a prior and
#           update it in the process
theta_0=0.5
b_collector=[]
theta_collector=[theta_0]
for i in range(n_sample+n_burn):
    #sample b
    cur_b=np.random.gamma(3,1/(1+theta_collector[-1]))
    b_collector.append(cur_b)
    #sample theta
    cur_theta=np.random.gamma(a+a_delta,1/(cur_b+b_delta))
    theta_collector.append(cur_theta)
#part a: density
plt.hist(theta_collector[n_burn+1:],bins=50)
plt.title('Histogram for theta')
plt.savefig('q3_parta_hist_theta_case1.pdf')
plt.show()
plt.hist(b_collector[n_burn:],bins=50)
plt.title('Histogram for b')
plt.savefig('q3_parta_hist_b_case1.pdf')
plt.show()
#part b: posterior mean
print('Posterior mean of theta is '+
      str(np.mean(theta_collector[n_burn+1:])))
print('Posterior mean of b is '+
      str(np.mean(b_collector[n_burn:])))
#part c: 95% equitailed credible interval
print('95% equitailed credible interval is given by ('+
      str(np.percentile(theta_collector,2.5))+', '+
      str(np.percentile(theta_collector,97.5))+')')




#second case: we assume the distribution of b is fixed/knowned without
#            uncertainty and we do not update it in the process
b_collector=[]
theta_collector=[]
for i in range(n_sample+n_burn):
    #sample b
    cur_b=np.random.exponential(1/lam)
    b_collector.append(cur_b)
    #sample theta
    cur_theta=np.random.gamma(a+a_delta,1/(cur_b+b_delta))
    theta_collector.append(cur_theta)
#part a: density
plt.hist(theta_collector[n_burn:],bins=50)
plt.title('Histogram for theta')
plt.savefig('q3_parta_hist_theta_case2.pdf')
plt.show()
plt.hist(b_collector[n_burn:],bins=50)
plt.title('Histogram for b')
plt.savefig('q3_parta_hist_b_case2.pdf')
plt.show()
#part b: posterior mean
print('Posterior mean of theta is '+
      str(np.mean(theta_collector[n_burn:])))
print('Posterior mean of b is '+
      str(np.mean(b_collector[n_burn:])))
#part c: 95% equitailed credible interval
print('95% equitailed credible interval is given by ('+
      str(np.percentile(theta_collector,2.5))+', '+
      str(np.percentile(theta_collector,97.5))+')')


