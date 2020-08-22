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
N = 100
height = np.random.normal(loc=10,scale=2,size=N)
leg_prop = np.random.uniform(size=N,low=0.4,high=0.5)
leg_left = leg_prop*height + np.random.uniform(size=N,low=0,high=0.02)
leg_right = leg_prop*height + np.random.uniform(size=N,low=0,high=0.02)


fig,ax = plt.subplots(nrows=2)
ax[0].plot(leg_right,height,'b.')
ax[1].plot(leg_left,height,'b.')
plt.show()

# Model with multicolinearity
with pm.Model() as model_collinear:
    a = pm.Normal('a', mu=10, sigma=100)
    bl = pm.Normal('bl', mu=2, sigma=10)
    br = pm.Normal('br', mu=2 ,sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bl*leg_left + br*leg_right)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=height)
    trace_collinear = pm.sample(cores=2)


# Model with no collinearity
#%%
with pm.Model() as model_no_collinear:
    a = pm.Normal('a', mu=10, sigma=100)
    br = pm.Normal('br', mu=2 ,sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + br*leg_right)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=height)
    trace_no_collinear = pm.sample(cores=2)

#%%
model_collinear.name='collinear'
model_no_collinear.name='no-collinear'
df_comp_models = pm.compare({model_collinear:trace_collinear,model_no_collinear:trace_no_collinear})
df_comp_models

#%%
pm.forestplot(trace_collinear,var_names=['a','bl','br','sigma'])
pm.forestplot(trace_no_collinear,var_names=['a','br','sigma'])


# Posterior predictive 
#%%
collinear_ppc = pm.sample_posterior_predictive(trace_collinear,samples=500,model=model_collinear)
no_collinear_ppc = pm.sample_posterior_predictive(trace_no_collinear,samples=500,model=model_no_collinear)

_,ax = plt.subplots(figsize=(12,6))
ax.hist([h.mean() for h in collinear_ppc['h']])
ax.axvline(height.mean())
ax.set(title='Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency')

_,ax = plt.subplots(figsize=(12,6))
ax.hist([h.mean() for h in no_collinear_ppc['h']])
ax.axvline(height.mean())
ax.set(title='Posterior predictive of the mean', xlabel='mean(x)', ylabel='Frequency')

# Plot posterior density for models
#%%
az.plot_density([trace_collinear, trace_no_collinear], 
data_labels=['collinear','no collinear'], 
var_names=['br'], 
shade=0.1)

# Compare plots
#%%
az.plot_forest([trace_collinear, trace_no_collinear], 
model_names=['collinear','no collinear'],
var_names=['br','a','sigma'])

#%%
az.plot_forest([trace_collinear, trace_no_collinear], 
model_names=['collinear','no collinear'],
var_names=['br','a','sigma'],
kind='ridgeplot')

