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
from PIL import Image


cities = [ {'city': "Bari", "italian": "Bari", 'region': "Puglia"}, 
           {'city': "Bologna", "italian": "Bologna", 'region': "Emilia-Romagna"},
           {'city': "Cagliari", "italian": "Cagliari", 'region': "Sardegna"},
           {'city': "Florence", "italian": "Firenze", 'region': "Toscana"},
           {'city': "Milan", "italian": "Milano", 'region': "Lombardia"},
           {'city': "Naples", "italian": "Napoli", 'region': "Campania"},
           {'city': "Palermo", "italian": "Palermo", 'region': "Sicilia"},
           {'city': "Rome", "italian": "Roma", 'region': "Lazio"},
           {'city': "Turin", "italian": "Torino", 'region': "Piemonte"},
           {'city': "Venice", "italian": "Venezia", 'region': "Veneto"}]

def getCities():
    list_of_cities = []
    for city in cities:
        list_of_cities.append(city['city'])
    return list_of_cities

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

    parser.add_argument(
        "--run_all",
        metavar="Run all cities",
        help="Executes over all 10 cities",
        action = "store_true",
        default=False,
        widget="CheckBox",
    )

    parser.add_argument(
        "--city",
        metavar="Choose city to run",
        widget="Dropdown",
        choices=getCities(),
    )

    return parser.parse_args()

sentiments = [{"sent":"fear", "c":"#ff5733"}, {"sent":"anger", "c":"#5ba15f"}, {"sent":"joy", "c":"#59A5D8"}]

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
        if (conf.city == city['city'] or conf.run_all == True):
            print(f"Preparing {city['city']}")

            directory = f"{conf.datadir}/{city['city']}/sentiment"

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

            print(f"{city['city']} had {len(df)} tweets")
            print(df['joy'].resample('M').mean()*100)
            print(df['fear'].resample('M').mean()*100)
            print(df['anger'].resample('M').mean()*100)
            print(df['joy'].resample('Y').mean()*100)
            print(df['fear'].resample('Y').mean()*100)
            print(df['anger'].resample('Y').mean()*100)

            filename = f"{conf.datadir}/{city['city']}/{city['italian']}.csv"
            citta = pd.read_csv(open(filename,'rb'))
            citta['DateTime'] = pd.to_datetime(citta['data'])
            citta = citta.set_index(['DateTime'])

            filename2 = f"{conf.datadir}/{city['city']}/{city['region']}-death.csv"
            dead = pd.read_csv(open(filename2,'rb'))
            dead['DateTime'] = pd.to_datetime(dead['data'])
            dead = dead.set_index(['DateTime'])

            # for sentiment in sentiments:
            color = 'black'
            fig, ax1 = plt.subplots(figsize=(30,15))
            plt.plot(dead.index, dead.Death, color = color,label=f'Daily Covid Deaths in the {city["region"]} region', alpha=0.6)
            plt.fill_between(dead.index, dead.Death, color = "#FFD300", alpha=0.6)
            plt.bar(citta.index, citta.Delta, color = "grey", alpha=0.6,label=f'Daily Covid Cases in {city["city"]}')

            ax1.set_xlabel('Date', size=15)
            ax1.set_ylabel('Daily Covid Cases & Deaths', color="black", size=15) 
            start, end = ax1.get_xlim()
            ax1.xaxis.set_ticks(np.arange(start, end, 0.1))
            ax1.tick_params(axis='y', labelcolor="black", labelsize=14)
            ax1.tick_params(axis='x', labelcolor="black", labelsize=14)
            ax1.set_ylim(bottom=0)

            weekend_indices = find_weekend_indices(citta.index)
            highlight_datetimes(weekend_indices, ax1, citta)
            
            # color = 'tab:red'
            ax2 = plt.twinx()
            
            for sentiment in sentiments:
                mean = df[sentiment['sent']].resample('D').mean()*100
                ax2.scatter(mean.index, mean, marker = '.', color = sentiment['c'])
                x_av = movingaverage(mean, 1)
                ax2.plot(mean.index, x_av, color = sentiment['c'], label=f"{sentiment['sent']} tweets")
            
            ax2.set_ylabel(f'Tweet Sentiment in (%)', color="black", size=15) 
            ax2.tick_params(axis='y', labelcolor="black", labelsize=14)
            ax2.set_ylim(bottom=0)
            
            bbox_props = dict(boxstyle="round", fc="white", alpha=0.8)

            annotations_file = f"{conf.datadir}/{city['city']}.xlsx"
            a_file = pd.read_excel(open(annotations_file,'rb'))

            for date, row in a_file.iterrows():
                ax2.annotate(row.Policy, (mdates.date2num(row.Date), mean[row.Date]), xytext=(row.X, row.Y), 
                            bbox=bbox_props, size=row.Size,textcoords='offset points', arrowprops=dict(arrowstyle='-|>'))
            
            ax1.legend(prop={'size': 15}, loc='upper left')
            ax2.legend(prop={'size': 15}, loc='upper right')
            ax1.set_title(label=f'Visualizing tweet sentiment in {city["city"]} during the time of COVID-19', size=21)
            fig.tight_layout()
            fig.savefig(f"{conf.datadir}/{city['city']}/{city['city']}.png", dpi=150)
            image = Image.open(f"{conf.datadir}/{city['city']}/{city['city']}.png")
            image.show()
            plt.close(fig)