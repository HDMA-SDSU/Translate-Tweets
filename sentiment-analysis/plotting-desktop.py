from gooey import Gooey, GooeyParser
import io
import sys
import json
import os
import string
from langdetect import detect
import re
from contextlib import redirect_stderr
import pickle
import pandas as pd
from collections import Counter
import seaborn as sns
from tqdm import tqdm as tqdm
from tqdm import trange
import datetime
import typing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from pylab import plot, ylim, xlim, show, xlabel, ylabel, grid
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
from wordcloud import WordCloud
from tqdm import tqdm as tqdm
from tqdm import trange


@Gooey(program_name="COVID-Crowdfight Sentiment Analysis", progress_regex=r"(\d+)%")
def parse_args():
    prog_descrip = "Apply sentiment analyis"

    parser = GooeyParser(description=prog_descrip)

    parser.add_argument(
        "datadir",
        metavar="Cleaned Twitter Data",
        help="Select the source twitter data folder",
        widget="DirChooser",
    )

    parser.add_argument(
        "city_name",
        metavar="City Name",
        help="Name of city to be analysed",
        type=str,
    )

    parser.add_argument(
        "--datacol",
        metavar="Data Column Name",
        help="Name of column where cleaned data is stored",
        type=str,
        default="cleaned_text",
    )

    parser.add_argument(
        "--mov_avg",
        metavar="Number of days for moving average",
        help="Number of days for moving average",
        type=int,
        default=3,
    )

    return parser.parse_args()

if __name__ == "__main__":
    # Initialise GUI
    conf = parse_args()

    # Read input data
    directory = conf.datadir

    meanArr = {"date":[], 'anger':[], 'anticipation':[], 'disgust':[], 'fear':[], 'joy':[], 'negative':[], 'positive':[], 'sadness':[], 'surprise':[], 'trust':[]}

    files = []

    # This is only for showing progress bar
    progress_bar_output = io.StringIO()

    print(f"Loading data")

    with redirect_stderr(progress_bar_output):
        for file in tqdm(os.listdir(directory), file=sys.stdout):
            filename = os.path.join(directory, file)
            if filename.endswith(".pickle") or filename.endswith(".pkl"):
                df = pickle.load(open(filename,'rb'))
                files.append(df)
            elif filename.endswith(".xlsx"):
                df = pd.read_excel(filename)
                files.append(df)
            else:
                raise NotImplementedError("File type not supported")
            print(progress_bar_output.read())

    print(f"Data loaded")

    for file in files:
        date = datetime.datetime.strptime(file['created_at'].min(), '%Y-%m-%d %H:%M:%S').date()
        
        meanArr["date"].append(date)
        meanArr["fear"].append(file['fear'].mean()*100)
        meanArr["anger"].append(file['anger'].mean()*100)
        meanArr["joy"].append(file['joy'].mean()*100)
        meanArr["positive"].append(file['positive'].mean()*100)
        meanArr["negative"].append(file['negative'].mean()*100)
        meanArr["anticipation"].append(file['anticipation'].mean()*100)
        meanArr["disgust"].append(file['disgust'].mean()*100)
        meanArr["sadness"].append(file['sadness'].mean()*100)
        meanArr["surprise"].append(file['surprise'].mean()*100)
        meanArr["trust"].append(file['trust'].mean()*100)
        
    print(f"Preparing plots")

    plt.figure(figsize=(20,8))
    plt.title(f'Change in fear, anger and joy for {conf.city_name}', fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Sentiment %', fontsize=15)
    plt.scatter(meanArr["date"], meanArr["fear"], marker = 'x', label='fear', color = '#FF8800')
    plt.scatter(meanArr["date"], meanArr["anger"], marker = 'o', label='anger', color = '#009900')
    plt.scatter(meanArr["date"], meanArr["joy"], marker = '.', label='joy', color = '#0080FF')


    def movingaverage(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')

    x_av_fear = movingaverage(meanArr["fear"], conf.mov_avg)
    x_av_joy = movingaverage(meanArr["joy"], conf.mov_avg)
    x_av_anger = movingaverage(meanArr["anger"], conf.mov_avg)
    plt.plot(meanArr["date"], x_av_fear, label=f'mov. avg. {conf.mov_avg}D fear', color = '#FF8800')
    plt.plot(meanArr["date"], x_av_joy, label=f'mov. avg. {conf.mov_avg}D joy', color = '#0080FF')
    plt.plot(meanArr["date"], x_av_anger, label=f'mov. avg. {conf.mov_avg}D anger', color = '#009900')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(directory, f"{conf.city_name}_sentiment_fear.png"))


    # positive / negative sentiment 
    plt.figure(figsize=(20,8))
    plt.title(f'Change in positive and negative sentiment for {conf.city_name}', fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Sentiment %', fontsize=15)
    plt.scatter(meanArr["date"], meanArr["positive"], marker = 'x', label='positive', color = '#FF8800')
    plt.scatter(meanArr["date"], meanArr["negative"], marker = 'o', label='negative', color = '#009900')

    # for emotion in emotions:
    #     plt.scatter(meanArr["date"], meanArr[emotion], marker = 'x')

    x_av_positive = movingaverage(meanArr["positive"], conf.mov_avg)
    x_av_negative = movingaverage(meanArr["negative"], conf.mov_avg)
    plt.plot(meanArr["date"], x_av_positive, label=f'mov. avg. {conf.mov_avg}D positive', color = '#FF8800')
    plt.plot(meanArr["date"], x_av_negative, label=f'mov. avg. {conf.mov_avg}D negative', color = '#009900')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(directory, f"{conf.city_name}_sentiment_posneg.png"))


    df = pd.concat(files)
    df.to_pickle(os.path.join(directory, f"{str(conf.city_name)}.pickle"))

    print(f"Preparing wordcloud")

    wordcloud = WordCloud(width=1920, height=1080).generate(' '.join(df[conf.datacol].astype(str)))
    wordcloud.to_file(os.path.join(directory, f"{conf.city_name}_wordcloud.png"))

    # DONE!
    print(f"Success")