from cProfile import label

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from bokeh.layouts import column
from statsmodels.tsa.seasonal import seasonal_decompose
import xarray as xr
import calendar


class PlotData:
    def __init__(self, column: str='lwp', ds: xr.Dataset = None):
        self.column = column
        self.ds = ds

    #'@staticmethod


    def temporalplot2 (self, column: str='lwp'):
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

    def temporalplot_grouped (self, column: str = 'lwp', groupby: str = 'monthly'):
        column_mean = self.ds[column].mean(dim=['lat', 'lon'])

        plt.figure(figsize=(12, 6))

        if groupby == 'monthly':
            # Calculate monthly averages across years
            monthly_avg = column_mean.groupby('time.month').mean()
            monthly_avg.plot()

            # Format x-axis with month names
            plt.xticks(range(1, 13), calendar.month_abbr[1:13])
            plt.title(f'Monthly Average {column}')
            plt.xlabel('Month')
        else:
            # Original time series plot
            column_mean.plot()
            plt.title(f'Mean {column} across time')
            plt.xlabel('Time')

        plt.ylabel(f'Mean {column}')
        plt.tight_layout()
        plt.show()

    def spatialplot(self, column: str='lwp'):
        time_avg = self.ds[column].mean(dim=['time'])
        plt.figure(figsize=(12, 6))
        time_avg.plot(x='lon', y='lat', robust=True)
        # sc = plt.scatter(df['lon'], df['lat'], c=df[column], cmap='viridis', alpha=0.7)
        # plt.colorbar(sc, label=column)
        plt.title(f"Spatial Distribution of {column} (Zoomed)")
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
    def ts_decompose(df, column: str):
        ts_decomp_column = seasonal_decompose(df[column], model='additive', period=12,
                                       extrapolate_trend='freq')  #period = 12) # we set the cyclic period of the seasonal cycle by hand
        trend_estimate = ts_decomp_column.trend
        seasonal_estimate = ts_decomp_column.seasonal
        residual_estimate = ts_decomp_column.resid

        # Plotting the time series and its individual components together
        fig, ax = plt.subplots(4, 1, sharex=True, sharey=False)
        #fig, ax = plt.subplots(5, 1, sharex=True, sharey=False)

        fig.set_figheight(10)
        fig.set_figwidth(20)

        ax[0].plot(df[column], label='Original')
        ax[0].legend(loc='upper left')

        ax[1].plot(trend_estimate, label='Trend')
        ax[1].legend(loc='upper left')

        ax[2].plot(seasonal_estimate, label='Seasonality')
        ax[2].legend(loc='upper left')

        ax[3].plot(residual_estimate, label='Residuals')
        ax[3].legend(loc='upper left')


    def ts_decompose_group(df, column: str, groupby='monthly'):
        # Ensure datetime index
        df = df.set_index('time')

        # Perform decomposition
        ts_decomp_column = seasonal_decompose(df[column], model='additive',
                                              period=12, extrapolate_trend='freq')

        # Extract components
        components = {
            'original': df[column],
            'trend': ts_decomp_column.trend,
            'seasonal': ts_decomp_column.seasonal,
            'residual': ts_decomp_column.resid
        }

        # Group by month if specified
        if groupby == 'monthly':
            components = {
                key: comp.groupby(comp.index.month).mean()
                for key, comp in components.items()
            }

        # Plot configuration
        fig, ax = plt.subplots(4, 1, figsize=(20, 10))
        titles = ['Original Series', 'Trend Component',
                  'Seasonal Component', 'Residual Component']

        for i, (key, title) in enumerate(zip(components.keys(), titles)):
            ax[i].plot(components[key] if not groupby else components[key].index,
                       components[key], label=title)

            if groupby == 'monthly':
                ax[i].set_xticks(range(1, 13))
                ax[i].set_xticklabels(calendar.month_abbr[1:13])
                ax[i].set_xlabel('Month')

            ax[i].legend(loc='upper left')
            ax[i].grid(True)

        plt.tight_layout()
        return fig

    def plot_data_extent_on_map(ds, title="Data Extent (Bounding Box)"):
        """
        Plots the extent (bounding box) of the data on a world map using the min/max lat/lon from the dataset.
        Args:
            ds (xarray.Dataset): The dataset containing 'lat' and 'lon' coordinates.
            title (str): Title for the plot.
        """
        # Extract min/max lat/lon
        min_lon = float(ds['lon'].min())
        max_lon = float(ds['lon'].max())
        min_lat = float(ds['lat'].min())
        max_lat = float(ds['lat'].max())

        # Create bounding box coordinates
        bbox_lons = [min_lon, max_lon, max_lon, min_lon, min_lon]
        bbox_lats = [min_lat, min_lat, max_lat, max_lat, min_lat]

        plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.LAND, edgecolor='black')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

        # Plot bounding box
        ax.plot(bbox_lons, bbox_lats, color='red', linewidth=2, marker='o', transform=ccrs.PlateCarree(), label='Study Area Extent')
        ax.legend()
        plt.title(title)
        plt.show()
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
