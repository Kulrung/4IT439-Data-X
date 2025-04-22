import os
import ast
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, r2_score, mean_squared_error, 
                             mean_absolute_error)
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold, 
                                     learning_curve)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, OneHotEncoder)
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              HistGradientBoostingRegressor, HistGradientBoostingClassifier)
from sklearn.inspection import permutation_importance
from pandas.errors import PerformanceWarning

# Ignore performance warnings
warnings.simplefilter(action='ignore', category=PerformanceWarning)

# Custom numerical imputer using a regression model
class NumericalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, model=HistGradientBoostingRegressor()):
        self.model = model
        self.numerical_columns = None
        self.models = {}

    def fit(self, X, y=None):
        X = X.copy()
        self.numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns

        for col in self.numerical_columns:
            X_features = X[self.numerical_columns].drop(columns=[col])
            mask = X[col].notna()
            X_train, y_train = X_features.loc[mask], X.loc[mask, col]

            if len(X_train) > 10:
                model = clone(self.model)
                model.fit(X_train, y_train)
                self.models[col] = model

        return self

    def transform(self, X):
        X_copy = X.copy()

        for col in self.numerical_columns:
            mask = X_copy[col].isna()
            if mask.sum() == 0:
                continue
            if col in self.models:
                X_features = X_copy[self.numerical_columns].drop(columns=[col])
                X_missing = X_features.loc[mask]
                predicted = self.models[col].predict(X_missing)
                X_copy.loc[mask, col] = predicted

        return X_copy


# Custom categorical imputer using most frequent value
class CategoricalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, model=HistGradientBoostingClassifier()):
        self.model = model
        self.categorical_columns = None

    def fit(self, X, y=None):
        X = X.copy()
        self.categorical_columns = X.select_dtypes(include=['object', 'category']).columns

        for col in self.categorical_columns:
            if pd.isnull(X[col]).sum() > 0:
                most_frequent = X[col].mode()[0]
                X[col].fillna(most_frequent, inplace=True)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.categorical_columns:
            if pd.isnull(X_copy[col]).sum() > 0:
                most_frequent = X_copy[col].mode()[0]
                X_copy[col].fillna(most_frequent, inplace=True)
        return X_copy


# Streamlit page configuration
st.set_page_config(page_title="Predikce cen Airbnb", layout="wide")
st.title('Prediction of Airbnb prices in Prague')

# Load pipeline and default values
with open("best_model_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)
with open("default_values.pkl", "rb") as f:
    default_values = pickle.load(f)

model = pipeline["model"]
scaler = pipeline["scaler"]

# Neighborhood options
neighbourhood_categories = [
    'Praha 1', 'Praha 3', 'Praha 2', 'Praha 7', 'Praha 4', 'Praha 14', 'Praha 5', 'Praha 8',
    'Praha 9', 'Praha 15', 'Praha 10', 'Praha 6', 'Praha 13', 'Velká Chuchle', 'Kunratice',
    'Zličín', 'Zbraslav', 'Petrovice', 'Praha 16', 'Praha 17', 'Praha 11', 'Praha 21',
    'Újezd', 'Slivenec', 'Ďáblice', 'Čakovice', 'Praha 18', 'Libuš', 'Řeporyje',
    'Praha 12', 'Nebušice', 'Satalice', 'Praha 22', 'Dolní Chabry', 'Praha 19',
    'Praha 20', 'Vinoř', 'Suchdol', 'Troja', 'Dubeč'
]

# User inputs
if model:
    st.header('Input parameters')
    col1, col2 = st.columns(2)
    user_input = {}

    with col1:
        user_input["bedrooms"] = st.number_input("Bedrooms", min_value=0, value=1, step=1)
        user_input["beds"] = st.number_input("Beds", min_value=1, value=1, step=1)
        user_input["bathrooms"] = st.number_input("Bathrooms", min_value=0, value=1, step=1)

    with col2:
        user_input["accommodates"] = st.number_input("Max number of guests", min_value=1, value=2, step=1)
        user_input["minimum_nights"] = st.number_input("Minimum nights", min_value=0, value=1, step=1)
        user_input["neighbourhood_cleansed"] = st.selectbox("Neighbourhood", options=neighbourhood_categories)

    df_input = pd.DataFrame([user_input])
    for column in default_values.keys():
        if column not in user_input:
            df_input[column] = default_values[column]

    df_input = df_input.drop(columns=["neighbourhood_cleansed"])
    for category in neighbourhood_categories:
        df_input[f"neighbourhood_cleansed_{category}"] = int(category == user_input["neighbourhood_cleansed"])

    numerical_columns = [
        'host_since', 'host_response_rate', 'host_acceptance_rate', 'host_listings_count',
        'host_total_listings_count', 'accommodates', 'bathrooms', 'bedrooms', 'beds',
        'minimum_nights', 'maximum_nights', 'minimum_minimum_nights', 'maximum_minimum_nights',
        'minimum_maximum_nights', 'maximum_maximum_nights', 'minimum_nights_avg_ntm',
        'maximum_nights_avg_ntm', 'availability_30', 'availability_60', 'availability_90',
        'availability_365', 'number_of_reviews', 'number_of_reviews_ltm', 'number_of_reviews_l30d',
        'availability_eoy', 'number_of_reviews_ly', 'first_review', 'last_review',
        'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
        'review_scores_value', 'calculated_host_listings_count',
        'calculated_host_listings_count_entire_homes', 'calculated_host_listings_count_private_rooms',
        'calculated_host_listings_count_shared_rooms', 'reviews_per_month']

    df_scaled = scaler.transform(df_input[numerical_columns])
    df_input_scaled = pd.DataFrame(df_scaled, columns=numerical_columns)
    df_combined = pd.concat([df_input_scaled, df_input.drop(columns=numerical_columns)], axis=1)

    predicted_price = model.predict(df_combined)[0]
    st.subheader(f"Predicted price: **{round(predicted_price)} Kč**")

# Sidebar info
st.sidebar.header('About this app')
st.sidebar.info('This app predicts the price of Airbnb accommodation in Prague based on data from recent listings.')
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Airbnb_Logo_B%C3%A9lo.svg/2560px-Airbnb_Logo_B%C3%A9lo.svg.png", width=200)