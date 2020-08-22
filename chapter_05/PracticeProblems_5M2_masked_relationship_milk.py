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
from sklearn import preprocessing

# Load waffle data_labels
df_milk = pd.read_csv('/home/amit/AmitKushwaha/Courses/Books/StatisticalRethinkingBook/Data/milk.csv',sep=';')
df_milk.dropna(inplace=True)

kcal_per_g = df_milk['kcal.per.g'].values
log_mass = np.log(df_milk['mass'].values)
neocortex_perc = df_milk['neocortex.perc'].values

fig,ax = plt.subplots(ncols=2)
ax[0].plot(log_mass,kcal_per_g,'b.')
ax[1].plot(neocortex_perc,kcal_per_g,'b.')
ax[0].set_xlabel('log.mass')
ax[1].set_xlabel('neocortex.perc')
ax[0].set_ylabel('mass')
plt.show()

#%%
# Model with single predictor 1
with pm.Model() as model_neocortex:
    a = pm.Normal('a', mu=0, sigma=100)
    bl = pm.Normal('bl', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', lower=0, upper=1)
    mu = pm.Deterministic('mu', a + bl*neocortex_perc)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=kcal_per_g)
    trace_neocortex = pm.sample(cores=2)

# Model with single predictor 2
with pm.Model() as model_log_mass:
    a = pm.Normal('a', mu=0, sigma=100)
    br = pm.Normal('br', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', lower=0, upper=1)
    mu = pm.Deterministic('mu', a + br*log_mass)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=kcal_per_g)
    trace_log_mass = pm.sample(cores=2)

# Model with both predictors
with pm.Model() as model_neocortex_mass:
    a = pm.Normal('a', mu=10, sigma=10)
    bl = pm.Normal('bl', mu=0, sigma=1)
    br = pm.Normal('br', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bl*neocortex_perc + br*log_mass)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=kcal_per_g)
    trace_neocortex_mass = pm.sample(cores=2)

#%%
# Plots
az.plot_forest([trace_neocortex_mass, trace_log_mass, trace_neocortex], 
model_names=['both_br_bl', 'only_br','only_bl'],
var_names=['bl','br'])

#%%
fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(10,10))

mu_mean = trace_neocortex['mu']
ax[0,0].scatter(neocortex_perc,kcal_per_g)
ax[0,0].plot(neocortex_perc, mu_mean.mean(0), 'C1')
az.plot_hpd(neocortex_perc, mu_mean,ax=ax[0,0])

mu_mean = trace_log_mass['mu']
ax[0,1].scatter(log_mass,kcal_per_g)
ax[0,1].plot(log_mass, mu_mean.mean(0), 'C1')
az.plot_hpd(log_mass, mu_mean,ax=ax[0,1])

seq = np.linspace(55, 76, 50)
mu_pred = trace_neocortex_mass['a'] + trace_neocortex_mass['bl'] * seq[:,None] + trace_neocortex_mass['br'] * log_mass.mean()
ax[1,0].plot(seq, mu_pred.mean(1), 'k')
az.plot_hpd(seq, mu_pred.T,
            fill_kwargs={'alpha': 0},
            plot_kwargs={'alpha':1, 'color':'k', 'ls':'--'},ax=ax[1,0])

ax[1,0].set_xlabel('neocortex.perc')
ax[1,0].set_ylabel('kcal.per.g')

seq = np.linspace(-3, 5, 50)
mu_pred = trace_neocortex_mass['a'] + trace_neocortex_mass['bl'] *neocortex_perc.mean() + trace_neocortex_mass['br'] *seq[:,None] 
ax[1,1].plot(seq, mu_pred.mean(1), 'k')
az.plot_hpd(seq, mu_pred.T,
            fill_kwargs={'alpha': 0},
            plot_kwargs={'alpha':1, 'color':'k', 'ls':'--'},ax=ax[1,1])

ax[1,1].set_xlabel('log.mass')
ax[1,1].set_ylabel('kcal.per.g')



#%%
az.summary(trace_neocortex, ['a','bl','sigma'], credible_interval=.89).round(3)

#%%
