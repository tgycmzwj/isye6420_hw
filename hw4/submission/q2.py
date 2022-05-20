import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
#define statistics
sum_t=512
n=20
c=3
d=1
alpha=100
beta=5

mu0=0.1

mu_collector=[mu0]
lambda_collector=[]
for i in range(51000):
    #sample lambda
    cur_lambda=np.random.gamma(n+c,1/(mu_collector[-1]*sum_t+alpha))
    lambda_collector.append(cur_lambda)
    #sample mu
    cur_mu=np.random.gamma(n+d,1/(lambda_collector[-1]*sum_t+beta))
    mu_collector.append(cur_mu)

mu_collector=mu_collector[1001:]
lambda_collector=lambda_collector[1000:]

print('Posterior mean of mu is '+str(np.mean(mu_collector)))
print('Posterior variance of mu is '+str(np.var(mu_collector)))

print('Posterior mean of lambda is '+str(np.mean(lambda_collector)))
print('Posterior variance of lambda is '+str(np.var(lambda_collector)))


print('The 95% equitailed credible set for mu is ('+
      str(np.percentile(mu_collector,2.5))+', '+
      str(np.percentile(mu_collector,97.5))+')')
print('The 95% equitailed credible set for lambda is ('+
      str(np.percentile(lambda_collector,2.5))+', '+
      str(np.percentile(lambda_collector,97.5))+')')
print('Bayes estimator of the product is given by '+
      str(np.mean([mu_collector[i]*lambda_collector[i]
                   for i in range(len(mu_collector))]))+')')
#scatter plot
plt.scatter(mu_collector,lambda_collector)
plt.title('Scatter plot of (mu,lambda)')
plt.xlabel('mu')
plt.ylabel('lambda')
plt.savefig('q2_partc_scatter.pdf')
plt.show()

plt.hist(mu_collector,bins=50)
plt.title('Histogram for mu')
plt.savefig('q2_partc_hist_mu.pdf')
plt.show()

plt.hist(lambda_collector,bins=50)
plt.title('Histogram for lambda')
plt.savefig('q2_partc_hist_lambda.pdf')
plt.show()














