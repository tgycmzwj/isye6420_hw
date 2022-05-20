import numpy as np
import scipy.stats as stats
from scipy.optimize import fsolve
from scipy.optimize import minimize
#question1
#part(b)
alpha=9/2
beta=4
theta_low1=stats.gamma.ppf(0.025,a=alpha,scale=1/beta)
theta_high1=stats.gamma.ppf(0.975,a=alpha,scale=1/beta)
print("for part (b), theta_low={}, theta_high={}".format(theta_low1,theta_high1))
#part(c)
def equations(theta):
    return [stats.gamma.pdf(theta[1],a=alpha,scale=1/beta)-stats.gamma.pdf(theta[0],a=alpha,scale=1/beta),
            stats.gamma.cdf(theta[1],a=alpha,scale=1/beta)-stats.gamma.cdf(theta[0],a=alpha,scale=1/beta)-0.95]
theta_low2,theta_high2=fsolve(equations,[0,100])
print("for part (c), theta_low={}, theta_high={}".format(theta_low2,theta_high2))
#part(d)
fun=lambda x: -stats.gamma.pdf(x,a=alpha,scale=1/beta)
results=minimize(fun,1)
print("for part (d), the mode is {} and the pdf at mode is {}".format(results.x[0],-results.fun))
#part(e)
posterior_odd=(1-stats.gamma.cdf(1,a=alpha,scale=1/beta))/stats.gamma.cdf(1,a=alpha,scale=1/beta)
