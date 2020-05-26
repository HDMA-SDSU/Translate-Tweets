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

    df = pd.concat(files)
    df.to_pickle(os.path.join(directory, f"{str(conf.city_name)}.pickle"))

    df['Datetime'] = pd.to_datetime(df['created_at'])
    df = df.set_index(['Datetime'])
    fig, ax = plt.subplots(figsize=(100,30))
    sns.set(font_scale=1) 
    sns_plot = sns.boxplot(df.index.dayofyear, df.fear*100, ax=ax, showfliers = False)
    fig = sns_plot.get_figure()
    fig.savefig(os.path.join(directory, f"{conf.city_name}_sentiment_fear_box.png"))


    print(f"Preparing plots")

    plt.figure(figsize=(20,8))
    plt.title(f'Change in fear, anger and joy for {conf.city_name}', fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Sentiment %', fontsize=15)
    fear_avg = df.fear.resample('D').mean()*100
    anger_avg = df.anger.resample('D').mean()*100
    joy_avg = df.joy.resample('D').mean()*100
    plt.scatter(fear_avg.index, fear_avg, marker = 'x', label='fear', color = '#FF8800')
    plt.scatter(anger_avg.index, anger_avg, marker = 'o', label='anger', color = '#009900')
    plt.scatter(joy_avg.index, joy_avg, marker = '.', label='joy', color = '#0080FF')

    def movingaverage(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')

    x_av_fear = movingaverage(fear_avg, conf.mov_avg)
    x_av_joy = movingaverage(joy_avg, conf.mov_avg)
    x_av_anger = movingaverage(anger_avg, conf.mov_avg)
    plt.plot(fear_avg.index, x_av_fear, label=f'mov. avg. {conf.mov_avg}D fear', color = '#FF8800')
    plt.plot(joy_avg.index, x_av_joy, label=f'mov. avg. {conf.mov_avg}D joy', color = '#0080FF')
    plt.plot(anger_avg.index, x_av_anger, label=f'mov. avg. {conf.mov_avg}D anger', color = '#009900')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(directory, f"{conf.city_name}_sentiment_fear.png"))


    # positive / negative sentiment 
    plt.figure(figsize=(20,8))
    plt.title(f'Change in positive and negative sentiment for {conf.city_name}', fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Sentiment %', fontsize=15)
    pos_avg = df.positive.resample('D').mean()*100
    neg_avg = df.negative.resample('D').mean()*100
    plt.scatter(pos_avg.index, pos_avg, marker = 'x', label='positive', color = '#FF8800')
    plt.scatter(neg_avg.index, neg_avg, marker = 'o', label='negative', color = '#009900')

    x_av_positive = movingaverage(pos_avg, conf.mov_avg)
    x_av_negative = movingaverage(neg_avg, conf.mov_avg)
    plt.plot(pos_avg.index, x_av_positive, label=f'mov. avg. {conf.mov_avg}D positive', color = '#FF8800')
    plt.plot(neg_avg.index, x_av_negative, label=f'mov. avg. {conf.mov_avg}D negative', color = '#009900')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(directory, f"{conf.city_name}_sentiment_posneg.png"))

    print(f"Preparing wordcloud")

    wordcloud = WordCloud(width=1920, height=1080).generate(' '.join(df[conf.datacol].astype(str)))
    wordcloud.to_file(os.path.join(directory, f"{conf.city_name}_wordcloud.png"))

    # DONE!
    print(f"Success")