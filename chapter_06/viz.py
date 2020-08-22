from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import arviz as az
az.style.use('arviz-darkgrid')


def plot_polynomial_regressions(d1,
                                trace_1,
                                xcol,
                                ycol,
                                degree=1,
                                credible_interval=.95,
                                plot_name='hpd',
                                ax=None):

    yval = d1[ycol].values
    # Polynomial features
    polynomial_features = PolynomialFeatures(degree=degree, include_bias=False)
    xval = polynomial_features.fit_transform(d1[xcol].values.reshape(-1, 1))

    if plot_name == 'hpd':
        if ax == None:
            fig, ax = plt.subplots(figsize=(6, 6))
        mu_mean = trace_1['mu']
        ax.scatter(xval[:, 0], yval)
        sorted_xval = np.argsort(xval[:, 0])
        ax.plot(xval[sorted_xval, 0], mu_mean.mean(0)[sorted_xval], 'C1')

        xseq = np.linspace(np.min(xval[:, 0]), np.max(xval[:, 0]), 50)
        xplot = polynomial_features.fit_transform(xseq.reshape(-1, 1)).T
        mu_pred = trace_1['a'].reshape(-1, 1) + np.dot(trace_1['b'], xplot)
        ax.plot(xseq, mu_pred.mean(0), 'k.')
        az.plot_hpd(xseq,
                    mu_pred,
                    fill_kwargs={'alpha': 0},
                    plot_kwargs={
                        'alpha': 1,
                        'color': 'k',
                        'ls': '--'
                    },
                    ax=ax,
                    credible_interval=credible_interval)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title('Polynomial fit with degree=%d' % degree)