
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

file = "./Daten/se_atlantic_df.csv"
df = pd.read_csv(file, index_col='time')
df

class PlotData:
    def __init__(self, dataframe: str, column: str):
        self.column = column
        self.dataframe = dataframe

    def temporal_plot(column):
        plt.figure(figsize=(15, 6))
        plt.plot(df.index, df[column], label=column, color='blue', alpha=0.7)
        plt.title(f"{column} Over Time")
        plt.xlabel('Time')
        plt.ylabel(column)
        plt.legend()
        plt.grid()
        plt.show()

    def spatial_plot(column):
        plt.figure(figsize=(10, 8))
        sc = plt.scatter(df['lon'], df['lat'], c=df[column], cmap='viridis', alpha=0.7)
        plt.colorbar(sc, label=column)
        plt.title(f"Spatial Distribution of {column}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid()
        plt.show()

    def spatial_plot_on_map(df, column, title, cmap, colorbar_label):
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





