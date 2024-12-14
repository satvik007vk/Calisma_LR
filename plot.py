import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from statsmodels.tsa.seasonal import seasonal_decompose

file = "./Daten/se_atlantic_df.csv"
df = pd.read_csv(file, index_col='time')
df

class PlotData:
    def __init__(self, dataframe: str, column: str):
        self.column = column
        self.dataframe = dataframe

    def temporal_plot(column: str):
        df.index = pd.to_datetime(df.index)

        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df[column], label=column, color='blue', alpha=0.7)
        plt.title(f"{column} Over Time")
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.legend()
        plt.grid()
        plt.show()

    def spatial_plot(column: str):
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(df['lon'], df['lat'], c=df[column], cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label=column)
        plt.title(f"Spatial Distribution of {column}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid()
        plt.show()

    def spatial_plot_on_map(df, column: str, title, cmap, colorbar_label):
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



    def ts_decompose(column: str):
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
