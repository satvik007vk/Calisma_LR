import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

file = "./Daten/se_atlantic_df.csv"
df = pd.read_csv(file)

df['time'] = pd.to_datetime(df['time'])

aggregated_df = df.groupby(['time', 'lat', 'lon']).median().reset_index()

print(f'Null values after aggregation to median\n{aggregated_df.isnull().sum()}')

# xarray conversion not needed
# ds = aggregated_df.set_index(['time','lat','lon']).to_xarray()

# Check for NULL values
print(f'Null Values:\n{aggregated_df.isnull().sum()}')

predictands = ['clf', 'lwp']  # Output variables
predictors = [var for var in aggregated_df.columns if var not in predictands + ['time', 'lat', 'lon']]  # Predictor variables

# ✅ Standard Scaler function ( works on pandas DataFrame instead of ds)
def scale_df(df, scalertype: str = 'standard', scale_predictands: bool = True) -> pd.DataFrame:
    if scalertype == 'standard':
        scaler = StandardScaler()
    elif scalertype == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError('scalertype must be "standard" or "minmax"')

    # Copy DataFrame to avoid modifying original data
    df_scaled = df.copy()

    # Scale predictors
    df_scaled[predictors] = scaler.fit_transform(df_scaled[predictors])

    # Scale predictands if required
    if scale_predictands:
        df_scaled[predictands] = scaler.fit_transform(df_scaled[predictands])

    return df_scaled

df_scaled = scale_df(aggregated_df, 'standard')

# Print mean and std of the scaled variables
for var in predictors:
    print(f" Standard Scaled: {var}: mean={df_scaled[var].mean():.2f}, std={df_scaled[var].std():.2f}")

# ✅ Correlation Check (Using pandas)
def compute_correlation(df_scaled, correlation_method: str = 'pearson'):
    correlation_method = correlation_method.lower()
    corr_matrix = df_scaled[predictors].corr(method=correlation_method)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title(f"Correlation Matrix ({correlation_method})")
    plt.show()

compute_correlation(df_scaled)

# ✅ Time-based split
df_scaled = df_scaled.sort_values('time')

# ✅ Get time values
time_values = df_scaled['time'].unique()

# ✅ Compute split index (80% train, 20% test)
split_index = int(len(time_values) * 0.80)
train_time = time_values[:split_index]
test_time = time_values[split_index:]

# ✅ Split dataset using pandas
df_train = df_scaled[df_scaled['time'].isin(train_time)]
df_test = df_scaled[df_scaled['time'].isin(test_time)]

# Print split summary
print(f"Training period: {train_time[0]} to {train_time[-1]}")
print(f"Testing period: {test_time[0]} to {test_time[-1]}")

print(f"Training data shape: {df_train.shape}")
print(f"Testing data shape: {df_test.shape}")

print("Checking missing values before saving:")
print(df_scaled.isnull().sum())

#TODO - find a way to access datasets without saving them as files

# ✅ Save train and test datasets as CSV
# df_train.to_csv("train_data.csv", index=False)
# df_test.to_csv("test_data.csv", index=False)
#
# print("Train and test datasets saved successfully!")
