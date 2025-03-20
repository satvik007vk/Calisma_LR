from cProfile import label

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from bokeh.layouts import column
from statsmodels.tsa.seasonal import seasonal_decompose
import xarray as xr


class PlotData:
    def __init__(self, column: str='lwp', ds: xr.Dataset = None):
        self.column = column
        self.ds = ds

    #'@staticmethod
    def temporalplot(self, column: str = 'lwp'):
        column_mean = self.ds[column].mean(dim=['lat', 'lon'])

        plt.figure(figsize=(12, 6))
        column_mean.plot()
        plt.title(f'Mean {column} across time')
        plt.xlabel('Time')
        plt.ylabel(f'Mean {column}')
        plt.show()

    def temporalplot2(self, column: str='lwp'):
        # Check and clean coordinate types
        try:
            time_coord = self.ds['time']
            if not np.issubdtype(time_coord.dtype, np.datetime64):
                # Attempt to convert 'time' to datetime if it's not already
                self.ds['time'] = pd.to_datetime(time_coord)
        except KeyError:
            raise ValueError("The dataset does not have a 'time' coordinate.")

        # Compute mean across latitude and longitude
        column_mean = self.ds[column].mean(dim=['lat', 'lon'])

        # Ensure the data to be plotted is valid
        if not np.issubdtype(column_mean.dtype, np.number):
            raise TypeError(
                f"The data type of the column '{column}' must be numeric for plotting. Found: {column_mean.dtype}")

        # Plotting
        plt.figure(figsize=(12, 6))
        column_mean.plot()
        plt.title(f"Mean {column} across time")
        plt.xlabel("Time")
        plt.ylabel(f"Mean {column}")
        plt.show()



    def spatialplot(self, column: str='lwp'):
        time_avg = self.ds[column].mean(dim=['time'])
        plt.figure(figsize=(12, 6))
        time_avg.plot(x='lon', y='lat', robust=True)
        # sc = plt.scatter(df['lon'], df['lat'], c=df[column], cmap='viridis', alpha=0.7)
        # plt.colorbar(sc, label=column)
        plt.title(f"Spatial Distribution of {column}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.tight_layout()
        plt.grid()
        plt.show()
    #

    def spatial_plot_on_map(df, column: str, title=None, cmap='viridis', colorbar_label=None):

        if title is None:
            title = f'Mean {column} '
        if colorbar_label is None:
            colorbar_label = column

        plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray', alpha=0.5)
        ax.add_feature(cfeature.OCEAN, edgecolor='black', facecolor='lightblue', alpha=0.3)

        # Scatter plot
        scatter = plt.scatter(
            df['lon'], df['lat'], c=df[column], cmap=cmap, s=10, alpha=0.7,
            transform=ccrs.PlateCarree()
        )
        plt.colorbar(scatter, label=colorbar_label, orientation="vertical", shrink=0.7)
        plt.title(title)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

    # Function to plot variables
    def plot_variable_on_map(ds, variable, title, time_agg='mean'):
        """
        Plots a variable from an xarray dataset on a world map.

        Parameters:
            ds (xarray.Dataset): The dataset containing the variable.
            variable (str): The name of the variable to plot.
            title (str): The title of the plot.
            time_agg (str): How to handle the time dimension. Options:
                            - 'mean': Aggregate the data by taking the mean over time.
                            - 'median': Aggregate the data by taking the median over time.
                            - 'time_idx': Plot a specific time index (e.g., 0 for the first time step).

        """
        # Create a figure
        plt.figure(figsize=(12, 6))

        # Create a map with PlateCarree projection
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()  # Set the map extent to global

        # Add features (coastlines, borders, etc.)
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

        # Handle the time dimension
        if time_agg == 'mean':
            # Take the time-mean of the variable
            data = ds[variable].mean(dim='time').values
        if time_agg == 'median':
            data = ds[variable].median(dim='time').values
        elif isinstance(time_agg, int):
            # Select a specific time index
            data = ds[variable].isel(time=time_agg).values
        else:
            raise ValueError("Invalid time_agg. Use 'mean' or a time index (int).")

        # Extract lon and lat
        lon = ds['lon'].values
        lat = ds['lat'].values

        # Use pcolormesh to plot the data (assuming regular grid)
        lon2d, lat2d = np.meshgrid(lon, lat)
        pc = ax.pcolormesh(
            lon2d, lat2d, data,
            cmap='viridis', transform=ccrs.PlateCarree()
        )

        # Add a colorbar
        cbar = plt.colorbar(pc, ax=ax, orientation='vertical', shrink=0.7)
        cbar.set_label(variable)

        # Set the title
        plt.title(title, fontsize=14)

        # Show the plot
        plt.show()
    # 
    # 
    # def ts_decompose(column: str):
    #     ts_decomp_column = seasonal_decompose(df[column], model='additive', period=12,
    #                                    extrapolate_trend='freq')  #period = 12) # we set the cyclic period of the seasonal cycle by hand
    #     trend_estimate = ts_decomp_column.trend
    #     seasonal_estimate = ts_decomp_column.seasonal
    #     residual_estimate = ts_decomp_column.resid
    # 
    #     # Plotting the time series and its individual components together
    #     fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)
    #     #fig, ax = plt.subplots(5, 1, sharex=True, sharey=False)
    # 
    #     fig.set_figheight(10)
    #     fig.set_figwidth(20)
    # 
    #     ax[0].plot(df[column], label='Original')
    #     ax[0].legend(loc='upper left')
    # 
    #     ax[1].plot(trend_estimate, label='Trend')
    #     ax[1].legend(loc='upper left')
    # 
    #     ax[2].plot(seasonal_estimate, label='Seasonality')
    #     ax[2].legend(loc='upper left')
    # 
    #     ax[3].plot(residual_estimate, label='Residuals')
    #     ax[3].legend(loc='upper left')


# #Calling functions and plotting
# file = "./Daten/se_atlantic_df.csv"
# df = pd.read_csv(file, index_col='time')
# df
#
# import xarray as xr
#
#
# # Step 2: Aggregate duplicate rows by taking the mean of each group
# df = df.groupby(['time', 'lon', 'lat']).mean().reset_index()
#
# # Step 3: Set the index of the DataFrame to 'time', 'lon', and 'lat'
# df.set_index(['time', 'lon', 'lat'], inplace=True)
#
# # Step 4: Convert the DataFrame to an xarray Dataset
# ds = xr.Dataset.from_dataframe(df)
#
# lwp_plot = PlotData(column='lwp', ds=ds)
# lwp_plot.temporalplot2(column='lwp')
