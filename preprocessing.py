import pandas as pd
from IPython.core.pylabtools import figsize
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

# #Outliers Removal
#
# y='lwp'
# #predictands = [y]
# #aggregated_df.plot(y=y, figsize=(10, 5))
# sns.boxplot(data=aggregated_df[predictands])
#
# plt.title(f"{predictands} before outliers removed")
# plt.show()

def remove_outliers_iqr(
        df: pd.DataFrame,
        columns=None,  # Require explicit column list (no ambiguous default)
        iqr_coef: float = 1.5  # Standard Tukey's fence coefficient
) -> pd.DataFrame:
    """
    Removes outliers using IQR method for specified columns.

    Args:
        df: Input DataFrame
        columns: List of columns to process
        iqr_coef: Multiplier for IQR (default 1.5 for moderate outliers)

    Returns:
        DataFrame with outliers removed
    """
#    predictands = ['clf','lwp']
    if columns is None:
        columns = predictands
    df_filtered = df.copy()

    for col in columns:
        Q1 = df_filtered[col].quantile(0.25)
        Q3 = df_filtered[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_coef * IQR
        upper_bound = Q3 + iqr_coef * IQR

        mask = df_filtered[col].between(lower_bound, upper_bound)
        df_filtered = df_filtered[mask]

    return df_filtered


def remove_outliers_percentile(
        df: pd.DataFrame,
        columns=None,
        lower_percentile: float = 0.03,
        upper_percentile: float = 0.97
) -> pd.DataFrame:
    """
    Removes outliers using percentile-based trimming for specified columns.

    Args:
        df: Input DataFrame
        columns: List of columns to process (default: ['clf', 'lwp'])
        lower_percentile: Lower cutoff percentile (e.g., 0.03 for 3rd percentile)
        upper_percentile: Upper cutoff percentile (e.g., 0.97 for 97th percentile)

    Returns:
        DataFrame with outliers removed
    """
#    predictands = ['clf','lwp']
    if columns is None:
        columns = predictands
    df_filtered = df.copy()

    for col in columns:
        lower_bound = df_filtered[col].quantile(lower_percentile)
        upper_bound = df_filtered[col].quantile(upper_percentile)
        mask = df_filtered[col].between(lower_bound, upper_bound)
        df_filtered = df_filtered[mask]

    return df_filtered

#
# # calling outlier removal function
# filtered_df = remove_outliers_iqr(
#     aggregated_df,
#     columns=predictands  # Or explicitly: columns=['clf', 'lwp']
# )
#
# aggregated_df=filtered_df
#
# y='lwp'
# #predictands=[y]
# #aggregated_df.plot(y=y, figsize=(10,5))
# sns.boxplot(data=aggregated_df[predictands])
# plt.title(f"{predictands} after outliers removed")
# plt.show()

# ✅ Standard Scaler function ( works on pandas DataFrame instead of ds)
def scale_df(df, scalertype: str = 'standard', scale_predictands: bool = True, predictors=None, predictands=None ) -> pd.DataFrame:
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

#df_scaled = scale_df(aggregated_df, 'standard')

# Print mean and std of the scaled variables
# for var in predictors:
#     print(f" Standard Scaled: {var}: mean={df_scaled[var].mean():.2f}, std={df_scaled[var].std():.2f}")

# ✅ Correlation Check (Using pandas)
def compute_correlation(df_scaled, correlation_method: str = 'pearson'):
    #df_scaled = scale_df(aggregated_df, 'standard')

    correlation_method = correlation_method.lower()
    corr_matrix = df_scaled[predictors].corr(method=correlation_method)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title(f"Correlation Matrix ({correlation_method})")
    plt.show()

# df_scaled = scale_df(aggregated_df, 'standard')
# compute_correlation(df_scaled=df_scaled)

#wrapper_function to load the preprocessed data in memory
def preprocess_data(filepath="./Daten/se_atlantic_df.csv",
                    scalertype='standard',
                    outlier_method='iqr',
                    scale_predictands=True,
                    selected_predictands=None):
    # Load and aggregate
    df = pd.read_csv(filepath)
    df['time'] = pd.to_datetime(df['time'])
    df = df.groupby(['time', 'lat', 'lon']).median().reset_index()

    # Define predictands based on input or use default
    if selected_predictands is None:
        predictands = ['clf', 'lwp']
    else:
        predictands = selected_predictands

    predictors = [col for col in df.columns if col not in predictands + ['time', 'lat', 'lon']]

    # Remove outliers
    if outlier_method == 'iqr':
        df = remove_outliers_iqr(df, columns=predictands)
    else:
        raise NotImplementedError(f"Outlier method '{outlier_method}' not implemented")

    # Scale
    df = scale_df(df, scalertype, scale_predictands, predictors, predictands)

    # Time-based train/test split
    df = df.sort_values('time')
    time_values = df['time'].unique()
    split_index = int(len(time_values) * 0.67)
    train_time = time_values[:split_index]
    test_time = time_values[split_index:]

    df_train = df[df['time'].isin(train_time)]
    df_test = df[df['time'].isin(test_time)]

    X_train = df_train[predictors]
    y_train = df_train[predictands]
    X_test = df_test[predictors]
    y_test = df_test[predictands]

    return df_train, df_test, X_train, y_train, X_test, y_test, predictors, predictands
#
# # ✅ Time-based split
# df_scaled = df_scaled.sort_values('time')
#
# # Get time values
# time_values = df_scaled['time'].unique()
#
# # Compute split index (80% train, 20% test)
# split_index = int(len(time_values) * 0.80)
# train_time = time_values[:split_index]
# test_time = time_values[split_index:]
#
# # Split dataset using pandas
# df_train = df_scaled[df_scaled['time'].isin(train_time)]
# df_test = df_scaled[df_scaled['time'].isin(test_time)]
#
# # Print split summary
# print(f"Training period: {train_time[0]} to {train_time[-1]}")
# print(f"Testing period: {test_time[0]} to {test_time[-1]}")
#
# print(f"Training data shape: {df_train.shape}")
# print(f"Testing data shape: {df_test.shape}")
#
# print("Checking missing values before saving:")
# print(df_scaled.isnull().sum())
#
# #TODO - find a way to access datasets without saving them as files
#
# #✅ Save train and test datasets as CSV
# # df_train.to_csv("train_data.csv", index=False)
# # df_test.to_csv("test_data.csv", index=False)
# #
# # print("Train and test datasets saved successfully!")
