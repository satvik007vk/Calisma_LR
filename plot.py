
import matplotlib.pyplot as plt
import pandas as pd

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
        pass




