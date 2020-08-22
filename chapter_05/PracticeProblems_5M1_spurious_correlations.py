## Problem 5M1
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
b_left = np.random.uniform(size=N,low=.4,high=.5)
b_right = np.random.uniform(size=N,low=.04,high=.05)
leg_left = b_left*height + np.random.uniform(size=N,low=0,high=0.02)
leg_right = b_right*height + np.random.uniform(size=N,low=0,high=0.02)

fig,ax = plt.subplots(ncols=2)
ax[0].plot(leg_right,height,'b.')
ax[1].plot(leg_left,height,'b.')
ax[0].set_xlabel('leg_right')
ax[1].set_xlabel('leg_left')
ax[0].set_ylabel('height')
plt.show()

# Model with single predictor 1
with pm.Model() as model_bl:
    a = pm.Normal('a', mu=10, sigma=100)
    bl = pm.Normal('bl', mu=2, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bl*leg_left)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=height)
    trace_bl = pm.sample(cores=2)

# Model with single predictor 2
with pm.Model() as model_br:
    a = pm.Normal('a', mu=10, sigma=100)
    br = pm.Normal('br', mu=2, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + br*leg_right)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=height)
    trace_br = pm.sample(cores=2)

# Model with both predictors
with pm.Model() as model_collinear:
    a = pm.Normal('a', mu=10, sigma=100)
    bl = pm.Normal('bl', mu=2, sigma=10)
    br = pm.Normal('br', mu=2 ,sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bl*leg_left + br*leg_right)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=height)
    trace_collinear = pm.sample(cores=2)

#%%
# Plots
az.plot_forest([trace_collinear, trace_br, trace_bl], 
model_names=['both_br_bl', 'only_br','only_bl'],
var_names=['bl','br'])



#%%
