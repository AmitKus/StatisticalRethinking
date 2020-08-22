#%%
import pymc3 as pm
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
import statsmodels.formula.api as smf

# Multicollinearity 
N = 120
height = np.random.normal(loc=10,scale=2,size=N)
leg_prop = np.random.uniform(size=N,low=0.4,high=0.5)
leg_left = leg_prop*height + np.random.uniform(size=N,low=0,high=0.02)
leg_right = leg_prop*height + np.random.uniform(size=N,low=0,high=0.02)
data = dict(leg_right=leg_right,height=height)

# Original version
with pm.Model() as model_orig:
    a = pm.Normal('a', mu=10, sigma=100)
    br = pm.Normal('br', mu=2 ,sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + br*leg_right)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=height)
    trace_orig = pm.sample(cores=2)

# GLM version
with pm.Model() as model1:
    priors = {"Intercept":pm.Normal.dist(mu=10, sigma=100), 
    "leg_right":pm.Normal.dist(mu=2 ,sigma=10),
    "sd":pm.Uniform.dist(lower=0, upper=10)}
    pm.glm.GLM.from_formula('height~leg_right', data, priors=priors)
    trace = pm.sample(cores=2)
    # prior = pm.sample_prior_predictive()
    posterior_predictive = pm.sample_posterior_predictive(trace)
    pm_data = az.from_pymc3(
        trace = trace,
        # prior = prior,
        posterior_predictive = posterior_predictive,
    )
    


#%%
# az.plot_ppc(pm_data, alpha=0.3)
# plt.show()

# #%%
# fig,ax = plt.subplots()
# ax.plot(leg_right,height,'r.')
# az.plot_hpd(leg_right,pm_data.posterior_predictive.to_array(), color='g', plot_kwargs={'ls':'--'})
# plt.show()


#%%
#%%
pm.forestplot(trace,var_names=['Intercept','leg_right','sd'])
pm.forestplot(trace_orig,var_names=['a','br','sigma'])