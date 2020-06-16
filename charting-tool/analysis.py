import os
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime as dt
import matplotlib.dates as mdates
from gooey import Gooey, GooeyParser
import io
import sys

@Gooey(program_name="COVID-Crowdfight Plotter", progress_regex=r"(\d+)%")
def parse_args():
    prog_descrip = "Prepare sentiment analysis plots"

    parser = GooeyParser(description=prog_descrip)

    parser.add_argument(
        "datadir",
        metavar="Twitter Data",
        help="Select the source twitter data folder (this will parse all pickle files)",
        widget="DirChooser",
    )

    return parser.parse_args()

cities = ["Bari", "Bologna", "Cagliari", "Florence", "Milan", "Naples", "Palermo", "Rome", "Turin", "Venice"]
italian_cities = ["Bari", "Bologna", "Cagliari", "Firenze", "Milano", "Napoli", "Palermo", "Roma", "Torino", "Venezia"]
italian_regions = ["Puglia", "Emilia-Romagna", "Sardegna", "Toscana", "Lombardia", "Campania", "Sicilia", "Lazio", "Piemonte", "Veneto"]
sentiments = ["fear", "anger", "joy"]

# cities = ["Milan"]
# italian_cities = ["Milano"]
# italian_regions = ["Lombardia"]
# sentiments = ["fear", "anger", "joy"]

counter = 0

def movingaverage(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')

def highlight_datetimes(indices, ax, df):
    i = 0
    while i < len(indices)-1:
        ax.axvspan(df.index[indices[i]], df.index[indices[i] + 1], facecolor='grey', edgecolor='none', alpha=.3)
        i += 1
        
def find_weekend_indices(datetime_array):
    indices = []
    for i in range(len(datetime_array)):
        if datetime_array[i].weekday() >= 5:
            indices.append(i)
    return indices



if __name__ == "__main__":
    # Initialise GUI
    conf = parse_args()

    for city in cities:
        print(f"Preparing {city}")

        directory = f"{conf.datadir}/{city}/sentiment"

        files = []

        print(f"Loading data")


        for file in os.listdir(directory):
            filename = os.path.join(directory, file)
            if filename.endswith(".pickle") or filename.endswith(".pkl"):
                df = pickle.load(open(filename,'rb'))
                files.append(df)
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(filename)
                files.append(df)
        #     else:
        #         raise NotImplementedError("File type not supported")

        print(f"Data loaded")

        df = pd.concat(files)

        df['Datetime'] = pd.to_datetime(df['created_at'])
        df = df.set_index(['Datetime'])

        # df.to_pickle(f"{conf.datadir}/{city}/{city}.pickle")

        print(f"{city} had {len(df)} tweets")
        print(df['joy'].resample('M').mean()*100)
        print(df['fear'].resample('M').mean()*100)
        print(df['anger'].resample('M').mean()*100)
        print(df['joy'].resample('Y').mean()*100)
        print(df['fear'].resample('Y').mean()*100)
        print(df['anger'].resample('Y').mean()*100)

        filename = f"{conf.datadir}/{city}/{italian_cities[counter]}.csv"
        citta = pd.read_csv(open(filename,'rb'))
        citta['DateTime'] = pd.to_datetime(citta['data'])
        citta = citta.set_index(['DateTime'])

        filename2 = f"{conf.datadir}/{city}/{italian_regions[counter]}-death.csv"
        dead = pd.read_csv(open(filename2,'rb'))
        dead['DateTime'] = pd.to_datetime(dead['data'])
        dead = dead.set_index(['DateTime'])

        for sentiment in sentiments:
            color = 'tab:blue'
            fig, ax1 = plt.subplots(figsize=(25,15))
            plt.bar(citta.index, citta.Delta, color = color, alpha=0.6,label=f'Daily new COVID-19 cases')
            plt.plot(dead.index, dead.Death, color = "black",label=f'Daily fatalities for the {italian_regions[counter]} region', alpha=0.6)
            plt.fill_between(dead.index, dead.Death, color = "black", alpha=0.6)
            ax1.set_xlabel('Date', size=13)
            ax1.set_ylabel('Dailty new COVID-19 cases', color=color, size=13) 
            start, end = ax1.get_xlim()
            ax1.xaxis.set_ticks(np.arange(start, end, 0.1))
            ax1.tick_params(axis='y', labelcolor=color, size=13)

            weekend_indices = find_weekend_indices(citta.index)
            highlight_datetimes(weekend_indices, ax1, citta)
            
            color = 'tab:red'
            mean = df[sentiment].resample('D').mean()*100
            ax2 = plt.twinx() 
            ax2.scatter(mean.index, mean, marker = '.', color = color)
            x_av = movingaverage(mean, 1)
            ax2.plot(mean.index, x_av, color = color, label=f'{sentiment}')
            ax2.set_ylabel(f'{sentiment} (%)', color=color, size=13) 
            ax2.tick_params(axis='y', labelcolor=color)
            
            bbox_props = dict(boxstyle="round", fc="white", alpha=0.8)

            annotations_file = f"{conf.datadir}/{city}.xlsx"
            a_file = pd.read_excel(open(annotations_file,'rb'))

            for date, row in a_file.iterrows():
                ax2.annotate(row.Policy, (mdates.date2num(row.Date), mean[row.Date]), xytext=(row.X, row.Y), 
                            bbox=bbox_props, size=row.Size,textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
            
            ax1.legend(prop={'size': 12}, loc='upper right')
            ax1.set_title(label=f'Visualizing the effects of {sentiment} in {city} during the time of COVID-19', size=15)
            fig.savefig(f"{conf.datadir}/{city}/{city}_{sentiment}.png")
            plt.close(fig)

        counter = counter + 1