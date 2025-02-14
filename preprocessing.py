import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


file = "./Daten/se_atlantic_df.csv"
df = pd.read_csv(file)

df['time'] = pd.to_datetime(df['time'])

aggregated_df=df.groupby(['time','lat','lon']).median().reset_index()
ds = aggregated_df.set_index(['time','lat','lon']).to_xarray()

#Check for NULL values
print(f' Null Values: \n {ds.to_dataframe().isnull().sum()} \n')

predictands = ['clf', 'lwp']  # Output variables
predictors = [var for var in ds.data_vars if var not in predictands]  # Predictor variables

#print(f'predictors: {predictors}')

#standard scaler

def scale_ds (scalertype: str='standard', scale_predictands: bool = True)-> xr.Dataset:

    #Standard Scale
    if scalertype == 'standard':
        scaler = StandardScaler()
    elif scalertype == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError('scalertype must be "standard" or "minmax"')

    # Standardize predictors
    ds_scaled = xr.Dataset(
        {var: (ds[var].dims, scaler.fit_transform(ds[var].values.reshape(-1, 1)).reshape(ds[var].shape))
         for var in predictors},
        coords=ds.coords
    )
    #

        # # Add predictands back without scaling
    for var in predictands:
        if scale_predictands == True:
            ds_scaled[var] = (ds[var].dims, scaler.fit_transform(ds[var].values.reshape(-1, 1)).reshape(ds[var].shape))
        else:
            ds_scaled[var] = ds[var]


    return ds_scaled

ds_scaled = scale_ds('standard')
# print mean and std of the scaled variables

for var in predictors:
    print(f" Standard Scaled: \n {var}: mean={ds_scaled[var].mean().item()}, std={ds_scaled[var].std().item()}")

# Correlation Check
def compute_correlation(correlation_method: str = 'pearson'):

    correlation_method = correlation_method.lower()
    df_scaled = ds_scaled.to_dataframe().reset_index()
    corr_matrix = df_scaled[predictors].corr(method=correlation_method)

    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title(f"Correlation Matrix ({correlation_method})")
    plt.show()

compute_correlation()

#TODO - Move data splitting to a different python file

# Ensure time is sorted
ds_scaled = ds_scaled.sortby('time')

# Get time values
time_values = ds_scaled['time'].values

# Compute split index (80% train, 20% test)
split_index = int(len(time_values) * 0.8)
train_time = time_values[:split_index]
test_time = time_values[split_index:]

# Split dataset
ds_train = ds_scaled.sel(time=train_time)
ds_test = ds_scaled.sel(time=test_time)

# Print split summary
print(f"Training period: {train_time[0]} to {train_time[-1]}")
print(f"Testing period: {test_time[0]} to {test_time[-1]}")

print(f"Train size: {ds_train.sizes}")
print(f"Test size: {ds_test.sizes}")

#TODO - make paths robust using pathlib
# ROOT = Path()
# root_dir = Path(__file__).resolve().parent
# save_dir = root_dir /'daten'/'created'
#Save train and test datasets
ds_train.to_netcdf("train_data.nc")
ds_test.to_netcdf("test_data.nc")


print("Train and test datasets saved successfully!")
