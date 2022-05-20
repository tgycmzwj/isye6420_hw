import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pandas as pd
from pymc.math import switch

if __name__ == "__main__":
    np.random.seed(123456)

    data = np.loadtxt("bladderc.csv", dtype=float, skiprows=1, delimiter=",")
    print(data.shape)
    time = np.array(
        [0, 1, 4, 7, 10, np.nan, 14, 18, np.nan, np.nan, 23, np.nan, np.nan, np.nan, np.nan, np.nan, 26, np.nan, np.nan,
         np.nan,
         29, 29, 29, np.nan, np.nan, np.nan, np.nan, 32, 34, 36, np.nan, 37, np.nan, np.nan, 41, np.nan, np.nan, np.nan,
         np.nan, np.nan, 49, np.nan, np.nan, np.nan,
         59, np.nan, np.nan, np.nan, 1, 1, np.nan, 9, 10, 13, np.nan, np.nan, 18, np.nan, np.nan, np.nan, 22, 25, 25,
         25, np.nan, np.nan, np.nan, np.nan, 38,
         np.nan, np.nan, np.nan, 41, 41, np.nan, 44, np.nan, 45, np.nan, 46, 49, 50, np.nan, 54, np.nan, 59])

    # non-nan is censored
    print(np.count_nonzero(~np.isnan(time)))  # censored:39, uncensored:47

    time_original = data[:, 0].copy()
    observed = data[:, 1].copy()  # y
    group = data[:, 2].copy()  # x

    # we need to separate the observed values and the censored values
    observed_mask = np.where(np.isnan(time))
    censored_mask = np.where(~np.isnan(time))
    censored = time[censored_mask]
    print(censored.shape)
    print(censored)

    y_uncensored = time_original[observed_mask]
    x_censored = group[censored_mask]
    x_uncensored = group[observed_mask]
    print(y_uncensored.shape, x_censored.shape, x_uncensored.shape)
    # (47,) (39,) (47,)
    print(y_uncensored)

    log2 = np.log(2)

    with pm.Model() as m:
        beta0 = pm.Normal("beta0", 1, tau=0.0001)
        beta1 = pm.Normal("beta1", 0, tau=0.0001)

        λ_censored = pm.math.exp(beta0 + beta1 * x_censored)

        λ_uncensored = pm.math.exp(beta0 + beta1 * x_uncensored)

        impute_censored = pm.Bound(
            pm.Exponential,
            lower=censored,
        )("impute_censored", lam=λ_censored, shape=censored.shape[0])

        likelihood = pm.Exponential(
            "likelihood",
            lam=λ_uncensored,
            observed=y_uncensored,
            shape=y_uncensored.shape[0],
        )

        mu0 = pm.Deterministic("mu0", (pm.math.exp(-beta0)))
        mu1 = pm.Deterministic("mu1", (pm.math.exp(-beta0 - beta1)))
        mudiff = pm.Deterministic("mudiff", mu1 - mu0)
        h1prob = pm.Deterministic("h1prob", mu1 > mu0)

        trace = pm.sample(
            10000, tune=2000, cores=4, init="auto", step=[pm.NUTS(target_accept=0.95)]
        )

    print("############### Problem 2 ###############")
    print(az.summary(trace, hdi_prob=0.9))