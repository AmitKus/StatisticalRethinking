# Medium Problems

## Problem 6M1  

### Write down and compare the definitions of AIC, DIC and WAIC. Which of these criteria is most general? Which assumptions are required to transform a more general criterion into a less general one?

**Akaike Information Criterion (AIC):** AIC is the most known information criterion. AIC provides a surprisingly simple estimate of the average out-of-sample deviance:

$AIC = D_{train} + 2p$
Where $p$ is the number of free parameters to be estimated in the model.

AIC provides an approximation of predictive accuracy, as measured by out-of-sample deviance. AIC is the oldest and most restrictive information criterion. AIC is an approximation that is reliable only when:

1. The priors are flat or overwhelmed by the liklihood.
2. The posterior distribution is approximately multivariate Gaussian.
3. The sample size $N$ is much greater than the number of parameters.

**Deviance Information Criterion (DIC):** DIC accommodates information priors, but still assumes that the posterior distribution in multivariate Gaussian and that $N >> k$.  It is particularly useful in Bayesian model selection problems where the posterior distributions of the models have been obtained by Markov chain Monte Carlo (MCMC) simulation

$DIC = \bar{D} + \left( \bar{D} - \hat{D}\right) = \bar{D} + p_D$

The difference $\bar{D} - \hat{D} = p_D$ is analogous to the number of parameters used in AIC. $D$  is the posterior distribution of deviance, $\bar{D}$ is the average of $D$ and $\hat{D}$ is the deviance calculated at the posterior mean.

**Widely Applicable Information Criterion (WAIC):** It does not require a multivariate Gaussian posterior, and it is often more accurate than DIC. The distinguishing feature of WAIC is that it is pointwise. It access flexibility of a model with respect to fitting each observation, and then sumps up across all observations.

$WAIC = -2 (\sum_{i=1}^N \log Pr(y_i) - \sum_{i=1}^N V(y_i))$

Where $V(y_i)$ is the variance in the log-likelihood for observation $i$ in the training sample.

## Learning: Statistics is no substitute for science.

## Problem 6M2  

### Explain the difference between model selection and model averaging. What information is lost under model selection? What information is lost under model averaging?

**Model Selection :** Choosing the model with the lowest AIC/DIC/WAIC value and then discarding others.
**Information lost :** Discards information about relative model accuracy contained in the differences among the AIC/DIC/WAIC values. Relative model accuracy provides advice about how confident we might be about models (conditional on the set of models compared.)
 
 **Model Comparison:** means using DIC/WAIC in combination with the estimates and posterior predictive checks from each model. It is just as important to understand why a model outperforms another as it is to measure the performance difference.

**Model Averaging:** means using DIC/WAIC to construct a posterior predictive distribution that exploits what we know about the relative accuracy of the models. This helps guard against overconfidence in model structure, in the same way that using the entire posterior distribution helps guard against overconfidence in parameter values.
**Information lost:** Interpretability of the models, data dredging can result in non-generalizable results.

## Problem 6M3 

### When comparing models with an information criterion, why must all models be fit to exactly the same observations? What would happen to the information criterion values, if the models were fit to different numbers of observations? Perform some experiments, if you are not sure.

Information criterion calculations involve summations which are not averaged by number of observations.

## Problem 6M4

### What happens to the effective number of parameters, as measured by DIC or WAIC, as a prior becomes more concentrated? Why? Perform some experiments, if you are not sure.

The effective number of parameters goes down.

## Problem 6M5

### Provide an informal explanation of why informative priors reduce overfitting.

Informative priors reduce the flexibility of the models, or in other words force the model to learn less from the data.

## Problem 6M6

### Provide an informal explanation of why overly informative priors result in underfitting.

Overly informative priors significantly constrain the model's flexibility.