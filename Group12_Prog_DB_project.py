import streamlit
import os
import sqlite3
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
import seaborn as sns
import tk
import PyQt5
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'WebAgg'
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
encoder = OneHotEncoder(handle_unknown = "ignore")
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lazypredict.Supervised import LazyRegressor
from lightgbm import LGBMRegressor 
import joblib
import mlflow
import dagshub

