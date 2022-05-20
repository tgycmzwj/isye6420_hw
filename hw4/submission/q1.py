import numpy as np
import matplotlib.pyplot as plt

#define statistics
np.random.seed(123456)
sum_x2=114.9707
sum_y2=105.9196
sum_xy=82.5247
n=100
#define density function
def f(rho):
    term1=(1-rho**2)**(-n/2-3/2)
    term2=1/(2*(1-rho**2))*(sum_x2-2*rho*sum_xy+sum_y2)
    return term1*np.exp(-term2) if (rho<=1 and rho>=-1) else 0

#sampling
rho0=0
results=[rho0]
for i in range(51000):
    new_rho=np.random.uniform(results[-1]-0.1,results[-1]+0.1)
    cut=min(f(new_rho)/f(results[-1]),1)
    if np.random.uniform(0,1)<cut:
        results.append(new_rho)
    else:
        results.append(results[-1])
results=results[1000:]
print('bayes estimator is '+str(np.mean(results)))
#histogram for all rhos after removing the first 1000 obs
plt.hist(results,bins=50)
plt.title('Distribution of rho U(rho-0.1,rho+0.1)')
plt.savefig('q1_partc1.pdf')
plt.show()

#realization of the last 1000 obs
plt.plot(results[-1000:])
plt.title('Last 1000 realizations U(rho-0.1,rho+0.1)')
plt.savefig('q1_partc2.pdf')
plt.show()


#sampling
rho0=0
results=[rho0]
for i in range(51000):
    new_rho=np.random.uniform(-1,1)
    cut=min(f(new_rho)/f(results[-1]),1)
    if np.random.uniform(0,1)<cut:
        results.append(new_rho)
    else:
        results.append(results[-1])
results=results[1000:]
print('bayes estimator is '+str(np.mean(results)))


#histogram for all rhos after removing the first 1000 obs
plt.hist(results,bins=50)
plt.title('Distribution of rho U(-1,+1)')
plt.savefig('q1_partd1.pdf')
plt.show()

#realization of the last 1000 obs
plt.plot(results[-1000:])
plt.title('Last 1000 realizations U(-1,+1)')
plt.savefig('q1_partd2.pdf')
plt.show()



















