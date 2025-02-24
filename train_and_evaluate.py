import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def load_train_test_data(predictand: str='both'):
    """ predictand = 'clf' or 'lwp' or 'both'   """

    df_train = pd.read_csv(ROOT_DIR / 'Daten' / 'train_test' / 'train_data.csv')
    df_test = pd.read_csv(ROOT_DIR / 'Daten' / 'train_test' / 'test_data.csv')

    # Drop missing values
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    print("Train and test data loaded successfully! Missing values dropped.")

    if predictand == 'both':
        predictand = ['clf', 'lwp']
        predictors = [var for var in df_train.columns if var not in predictand + ['time', 'lat', 'lon']]

    elif predictand == 'clf':
        predictand = ['clf']
        predictors = [var for var in df_train.columns if var not in predictand + ['time', 'lat', 'lon']+['lwp']]


    elif predictand == 'lwp':
        predictand = ['lwp']
        predictors = [var for var in df_train.columns if var not in predictand + ['time', 'lat', 'lon']+['clf']]
    #
    # predictors = [var for var in df_train.columns if var not in predictand + ['time', 'lat', 'lon']]

    # Split into X and y
    X_train = df_train[predictors]
    y_train = df_train[predictand]
    X_test = df_test[predictors]
    y_test = df_test[predictand]

    #TODO - Fix the return variables
    return df_train, df_test, X_train, y_train, X_test, y_test, predictors, predictand

def train_mlr(X_train, y_train):
    mlr_model  = LinearRegression()
    mlr_model.fit(X_train, y_train)
    return mlr_model

def train_rf(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_xgboost(X_train, y_train):
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"{model_name} MSE: {mse}")
    print(f"{model_name} RMSE: {rmse}")

    return mse, rmse, y_pred

def get_mlr_coefficients(mlr_model, predictors, predictands):
    mlr_coefficients_df = pd.DataFrame(mlr_model.coef_, index=predictands, columns=predictors)
    print(f"\nMLR Coefficients:\n{mlr_coefficients_df}")
    return mlr_coefficients_df

def shap_explainability(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)


if __name__ == "__main__":
    # Load Data
    df_train, df_test, X_train, y_train, X_test, y_test, predictors, predictands = load_train_test_data(predictand='lwp')

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    # Train Models
    mlr_model = train_mlr(X_train, y_train)

    # Evaluate Models
    evaluate_model(mlr_model, X_test, y_test, "Multi-Linear Regression")

    # Get Coefficients
    mlr_coefficients = get_mlr_coefficients(mlr_model, predictors, predictands)
