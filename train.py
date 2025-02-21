import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
import matplotlib.pyplot as plt
from pathlib import Path

ROOT_DIR = Path(__file__).parent

# Load train and test datasets
df_train = pd.read_csv(ROOT_DIR/'Daten'/'train_test'/'train_data.csv')
df_test = pd.read_csv(ROOT_DIR/'Daten'/'train_test'/'test_data.csv')

# # Convert to pandas DataFrame
# df_train = ds_train.to_dataframe().reset_index()
# df_test = ds_test.to_dataframe().reset_index()

print("Train and test data loaded successfully!")

# Define predictors (X) and targets (y)
predictands = ['clf', 'lwp']  # Variables we want to predict
predictors = [var for var in df_train.columns if var not in predictands + ['time', 'lat', 'lon']]

# Split into X and y
X_train = df_train[predictors]
y_train = df_train[predictands]
X_test = df_test[predictors]
y_test = df_test[predictands]

# Check for missing values
print(f'Train NaNs:\n{y_train.isna().sum()}')
print(f'Test NaNs:\n{y_test.isna().sum()}')

#mlr_model = M

# Initialize and train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions and evaluation for Random Forest
# y_pred_rf = rf_model.predict(X_test)
# mse_rf = mean_squared_error(y_test, y_pred_rf)
# print(f"Random Forest MSE: {mse_rf}") #OUTPUT - Random Forest MSE: 0.5692762548992099


# Initialize and train XGBoost model
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions and evaluation for XGBoost
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print(f"XGBoost MSE: {mse_xgb}") #OUPUT - XGBoost MSE: 0.5542984081943716

# SHAP Explainability for XGBoost
explainer = shap.Explainer(xgb_model, X_train)
shap_values = explainer(X_test)

# SHAP Summary Plot
shap.summary_plot(shap_values, X_test)
