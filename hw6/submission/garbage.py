
def cal_r2(beta, x, y, x_impute, y_impute):
    mis_pos_x = np.where(np.sum(x.mask, axis=1))[0][0]
    mis_pos_y = np.where(y.mask)[0][0]
    imputed_x = np.concatenate((x.data[:mis_pos_x], [[1, x_impute[0]]], x.data[mis_pos_x + 1:]))
    imputed_y = np.array(list(y.data[:mis_pos_y]) + list(y_impute) + list(y.data[mis_pos_y + 1:]))
    pred_y = beta[0] * imputed_x[:, 0] + beta[1] * imputed_x[:, 1] + beta[2]*imputed_x[:,1]**2
    y_m = np.mean(imputed_y)
    ess = np.sum((pred_y - y_m) ** 2)
    rss = np.sum((pred_y - imputed_y) ** 2)
    r2 = 1 - rss / (rss + ess)
    return r2
r2_collection2 = []
select_index = np.random.randint(0, trace2.posterior.beta2.shape[0] * trace2.posterior.beta2.shape[1], 5000)
all_beta, all_y_impute, all_x_impute = np.concatenate(trace2.posterior.beta2), np.concatenate(
    trace2.posterior.likelihood2_missing), np.concatenate(trace2.posterior.x_imputed2_missing)
for i in range(select_index.shape[0]):
    r2_collection2.append(cal_r2(all_beta[i], x, y, all_x_impute[i], all_y_impute[i]))
print(np.mean(r2_collection2))

# with pm.Model() as m:
#     mu_beta = pm.Normal("mu_beta", 0, tau=1e-6)
#     tau_beta=pm.Gamma('tau_beta', 0.001, 0.001)
#     beta = pm.Normal("beta", mu_beta, tau=tau_beta, shape=x.shape[1]+1)
#     x_imputed = pm.TruncatedNormal("x_imputed", mu=60, sigma=10, lower=0, observed=x)
#     x2_imputed = pm.math.sqr(x_imputed)
#     tau_lld = pm.Gamma('tau_lld', 0.001, 0.001)
#     lld = pm.Normal("likelihood", beta[0]*x_imputed[:,0]+beta[1]*x_imputed[:,1]+beta[2]*x2_imputed[:,1]**2, tau=tau_lld, observed=y, shape=y.shape[0])
#     trace = pm.sample(5000,tune=1000,cores=4,init="auto",step=[pm.NUTS(target_accept=.95)])
# with m:
#     print(az.summary(trace, hdi_prob=0.95))
#     az.summary(trace, hdi_prob=0.95).to_csv('q1_part2.csv')
#     # calculate r2
#     r2_collector = []
#     all_y, all_tau = np.concatenate(trace.posterior.lld_missing), np.concatenate(trace.posterior.tau_lld)
#     for i in range(len(all_y)):
#         imputed_y_vector = np.concatenate((y[:mp_y], np.array(all_y[i]), y[mp_y + 1:]))
#         sst = np.sum((imputed_y_vector - np.mean(imputed_y_vector)) ** 2)
#         sse = (10-3)/all_tau[i]
#         r2_collector.append(1-sse/sst)
#     print('r2 is given by: ' + str(np.mean(r2_collector)))
#
#
# print('part 2 finished')
