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
df_waffle_divorce = pd.read_csv('/home/amit/AmitKushwaha/Courses/Books/StatisticalRethinkingBook/Data/WaffleDivorce.csv',sep=';')
scaler = preprocessing.StandardScaler()
Marriage = scaler.fit_transform(df_waffle_divorce['Marriage'].values.reshape(-1,1))
MedianAgeMarriage = scaler.fit_transform(df_waffle_divorce['MedianAgeMarriage'].values.reshape(-1,1))
Divorce = df_waffle_divorce['Divorce'].values.reshape(-1,1)

fig,ax = plt.subplots(ncols=2)
ax[0].plot(Marriage,Divorce,'b.')
ax[1].plot(MedianAgeMarriage,Divorce,'b.')
ax[0].set_xlabel('Marriage')
ax[1].set_xlabel('MedianAgeMarriage')
ax[0].set_ylabel('Divorce')
plt.show()

#%%
# Model with single predictor 1
with pm.Model() as model_bl:
    a = pm.Normal('a', mu=10, sigma=100)
    bl = pm.Normal('bl', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bl*MedianAgeMarriage)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=Divorce)
    trace_bl = pm.sample(cores=2)

# Model with single predictor 2
with pm.Model() as model_br:
    a = pm.Normal('a', mu=10, sigma=100)
    br = pm.Normal('br', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + br*Marriage)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=Divorce)
    trace_br = pm.sample(cores=2)

# Model with both predictors
with pm.Model() as model_collinear:
    a = pm.Normal('a', mu=10, sigma=10)
    bl = pm.Normal('bl', mu=0, sigma=1)
    br = pm.Normal('br', mu=0, sigma=1)
    sigma = pm.Uniform('sigma', lower=0, upper=10)
    mu = pm.Deterministic('mu', a + bl*MedianAgeMarriage + br*Marriage)
    h = pm.Normal('h', mu=mu, sigma=sigma, observed=Divorce)
    trace_collinear = pm.sample(cores=2)

#%%
# Plots
az.plot_forest([trace_collinear, trace_br, trace_bl], 
model_names=['both_br_bl', 'only_br','only_bl'],
var_names=['bl','br'])

### Observations:
#Coefficient of Marriage br was positive for model just based on Marriage but 
# became close to zero when the MedianAgeMarriage was incoporated in the model. 
# The correlation between Marriage and Divorce was a spurious correlation.