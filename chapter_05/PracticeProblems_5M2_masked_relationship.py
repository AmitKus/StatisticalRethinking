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

# Masked relationship
N = 100
rho = -0.1
x_pos = np.random.normal(size=N)
x_neg = np.random.normal(loc=rho * x_pos, scale=np.sqrt(1 - rho**2), size=N)
y = np.random.normal(size=N, loc=x_pos - x_neg)

fig, ax = plt.subplots(ncols=2)
ax[0].plot(x_pos, y, 'b.')
ax[1].plot(x_neg, y, 'b.')
ax[0].set_xlabel('x_pos')
ax[1].set_xlabel('x_neg')
ax[0].set_ylabel('y')
plt.show()

#%%
# Model with single predictor 1
with pm.Model() as model_bl:
    a = pm.Normal('a', mu=10, sigma=100)
    bl = pm.Normal('bl', mu=2, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bl * x_pos)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=y)
    trace_bl = pm.sample(cores=2)

# Model with single predictor 2
with pm.Model() as model_br:
    a = pm.Normal('a', mu=10, sigma=100)
    br = pm.Normal('br', mu=2, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + br * x_neg)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=y)
    trace_br = pm.sample(cores=2)

# Model with both predictors
with pm.Model() as model_collinear:
    a = pm.Normal('a', mu=10, sigma=100)
    bl = pm.Normal('bl', mu=2, sigma=10)
    br = pm.Normal('br', mu=2, sigma=10)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bl * x_pos + br * x_neg)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=y)
    trace_collinear = pm.sample(cores=2)

#%%
# Plots
az.plot_forest([trace_collinear, trace_br, trace_bl],
               model_names=['both_br_bl', 'only_br', 'only_bl'],
               var_names=['bl', 'br'])

#%%
