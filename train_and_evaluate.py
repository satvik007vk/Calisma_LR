import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

from preprocessing import preprocess_data

ROOT_DIR = Path(__file__).resolve().parent


def load_train_test_data(predictands: str='both'):
    """ predictand = 'clf' or 'lwp' or 'both'   """

    # if predictands = 'both':
    #     predictands = ['clf', 'lwp']
    # else:
    #     predictands = [predictand]
    #
    #


    # train_csv = 'train_data_1.csv'
    # test_csv = 'test_data_1.csv'
    #
    # df_train = pd.read_csv(ROOT_DIR / 'Daten' / 'train_test' / train_csv)
    # df_test = pd.read_csv(ROOT_DIR / 'Daten' / 'train_test' / test_csv)

    df_train, df_test, X_train, y_train, X_test, Y_test, predictors, predictands = preprocess_data(scale_predictands=False)

    # Drop missing values
    df_train.dropna(inplace=True)
    df_test.dropna(inplace=True)

    print("Train and test data loaded successfully! Missing values dropped.")
    #
    # if predictands == 'both':
    #     predictands = ['clf', 'lwp']
    #     predictors = [var for var in df_train.columns if var not in predictands + ['time', 'lat', 'lon']]
    #
    # elif predictands == 'clf':
    #     predictands = ['clf']
    #     predictors = [var for var in df_train.columns if var not in predictands + ['time', 'lat', 'lon']+['lwp']]
    #
    #
    # elif predictands == 'lwp':
    #     predictands = ['lwp']
    #     predictors = [var for var in df_train.columns if var not in predictands + ['time', 'lat', 'lon']+['clf']]
    # #
    # # predictors = [var for var in df_train.columns if var not in predictand + ['time', 'lat', 'lon']]

    # Split into X and y
    X_train = df_train[predictors]
    y_train = df_train[predictands]
    X_test = df_test[predictors]
    y_test = df_test[predictands]


    return df_train, df_test, X_train, y_train, X_test, y_test, predictors, predictands

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

def train_mlp_regressor(X_train, y_train):
    mlp_regressor_model = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
    mlp_regressor_model.fit(X_train, y_train)
    return mlp_regressor_model

def train_multioutput_regressor (X_train, y_train, regressor_model ):

    if regressor_model == 'random_forest':
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    elif regressor_model == 'xgboost':
        regressor = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    elif regressor_model == 'mlp':
        regressor = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
    elif regressor_model == 'lr':
        regressor = MultiOutputRegressor()
    
    multioutput_regressor_model = MultiOutputRegressor(estimator=regressor)

    multioutput_regressor_model.fit(X_train, y_train)
    return multioutput_regressor_model

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2_scores = r2_score(y_test, y_pred)

    print(f"{model_name} \n MSE: {mse} \n RMSE: {rmse} \n R2: {r2_scores}")

    return mse, rmse, r2_scores,

def get_mlr_coefficients(mlr_model, predictors, predictands):
    mlr_coefficients_df = pd.DataFrame(mlr_model.coef_, index=predictands, columns=predictors)
    print(f"\nMLR Coefficients:\n{mlr_coefficients_df}")
    return mlr_coefficients_df

def shap_explainability(model, X_train, X_test):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)


# if __name__ == "__main__":
#     # Load Data
#     df_train, df_test, X_train, y_train, X_test, y_test, predictors, predictands = load_train_test_data(predictand='lwp')
#
#     print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
#
#     # Train Models
#     mlr_model = train_mlr(X_train, y_train)
#
#     # Evaluate Models
#     evaluate_model(mlr_model, X_test, y_test, "Multi-Linear Regression")
#
#     # Get Coefficients
#     mlr_coefficients = get_mlr_coefficients(mlr_model, predictors, predictands)
