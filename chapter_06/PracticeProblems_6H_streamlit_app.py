# import ptvsd
# ptvsd.enable_attach(address=('localhost', 5678))
# ptvsd.wait_for_attach(
# )  # Only include this line if you always wan't to attach the debugger

import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pymc3 as pm
import theano.tensor as tt
import pickle
from sklearn.preprocessing import PolynomialFeatures
from dataio import load_data
from model import Model
from util import save_model_trace, load_model_trace
from viz import plot_polynomial_regressions


def main():
    st.title('Bayesian Model Comparison')\

    # Load data
    d1, d2 = load_data(1000)

    page = st.sidebar.selectbox("Choose a page",
                                ['Data', 'Fit Models', 'Plot Results'])

    if page == 'Data':
        if st.checkbox('Show raw train data'):
            st.write(d1)
        if st.checkbox('Show raw test data'):
            st.write(d2)
    elif page == 'Fit Models':
        fit_all_models(d1)
    elif page == 'Plot Results':
        if st.checkbox('Compare HPD plots for all polynomial models'):
            plot_hpd_plots(d1)
        if st.checkbox('Plot polynomial models with degree'):
            plot_hpd_plots_with_degree(d1)
        if st.checkbox('Compare WAIC of polynomial models'):
            if st.checkbox('weak prior on std(b)'):
                compare_waic_for_models('weak')
            if st.checkbox('strong prior on std(b)'):
                compare_waic_for_models('strong')


def fit_all_models(d1):
    for degree in range(1, 7):
        model = Model(degree, xcol='age', ycol='height')
        # priors for std dev of b
        prior_b_std = [5, 100]
        for prior_b_sigma in prior_b_std:
            M1, trace1 = model.fit(d1, prior_b_sigma=prior_b_sigma)
            save_model_trace(
                'chapter_06/fitted_models/M%d_b_std_%d.pkl' %
                (degree, prior_b_sigma), M1, trace1)


def plot_hpd_plots(d1):
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(24, 18))
    degree = 1
    for row in range(0, 3):
        for col in range(0, 2):
            M1, trace1 = load_model_trace(
                filename='chapter_06/fitted_models/M%d.pkl' % degree)
            plot_polynomial_regressions(d1,
                                        trace1,
                                        xcol='age',
                                        ycol='height',
                                        degree=degree,
                                        ax=ax[row, col])
            degree += 1
    st.pyplot()


def plot_hpd_plots_with_degree(d1):
    degree = st.number_input("Degree of polynomial to fit",
                             value=1,
                             min_value=1,
                             max_value=6,
                             format='%d')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))

    # priors for std dev of b
    prior_b_std = [5, 100]
    for ind, prior_b_sigma in enumerate(prior_b_std):
        M1, trace1 = load_model_trace(
            'chapter_06/fitted_models/M%d_b_std_%d.pkl' %
            (degree, prior_b_sigma))

        plot_polynomial_regressions(d1,
                                    trace1,
                                    xcol='age',
                                    ycol='height',
                                    degree=degree,
                                    ax=ax[ind])
        ax[ind].set_title('Prior Std for b = %d' % prior_b_sigma)
    st.pyplot()


def compare_waic_for_models(prior_type):

    model_trace_dict = {}
    if prior_type == 'weak':
        prior_b_std = [100]
    else:
        prior_b_std = [5]
    for degree in range(1, 7):
        for prior_b_sigma in prior_b_std:
            model, trace = load_model_trace(
                'chapter_06/fitted_models/M%d_b_std_%d.pkl' %
                (degree, prior_b_sigma))
            model.name = 'M%d_b_std_%d.pkl' % (degree, prior_b_sigma)
            model_trace_dict[model.name] = trace

    df_comp_WAIC = pm.compare(model_trace_dict)
    st.table(
        df_comp_WAIC.style.format({
            'waic': '{:.2f}',
            'p_waic': '{:.2f}',
            'd_waic': '{:.2f}',
            'weight': '{:.2f}',
            'se': '{:.2f}',
            'dse': '{:.2f}'
        }))
    fig, ax = plt.subplots(figsize=(6, 6))
    pm.compareplot(df_comp_WAIC)
    st.pyplot()


if __name__ == '__main__':
    main()