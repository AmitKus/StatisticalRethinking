import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

DATE_COLUMN = 'date/time'
DATA_CSV = ('Data/Howell1.csv')


@st.cache
def load_data(nrows):
    howell1 = pd.read_csv(DATA_CSV, nrows=nrows, sep=';')

    # Normalize the age column
    howell1['age'] = StandardScaler().fit_transform(
        howell1['age'].values.reshape(-1, 1))

    # Divide dataframe into two equal
    d1, d2 = train_test_split(howell1, test_size=0.5, random_state=42)
    return d1, d2