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
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm as tqdm
from tqdm import trange
from nltk.tokenize.casual import TweetTokenizer
import datetime
import typing
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from pylab import plot, ylim, xlim, show, xlabel, ylabel, grid
from numpy import linspace, loadtxt, ones, convolve
import numpy as np
from wordcloud import WordCloud


stopcorpus: typing.List = stopwords.words('english')

@Gooey(program_name="COVID-Crowdfight Sentiment Analysis", progress_regex=r"(\d+)%")
def parse_args():
    prog_descrip = "Apply sentiment analyis"

    parser = GooeyParser(description=prog_descrip)

    parser.add_argument(
        "datadir",
        metavar="Translated Twitter Data",
        help="Select the source twitter pickled data folder",
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
        metavar="Output Data Column Name",
        help="Name of column where translated data is stored",
        type=str,
        default="translated_full_text",
    )

    return parser.parse_args()

def remove_links(text):
    import re
    return re.sub(r"http\S+", "", text)

def style_text(text:str):
    return text.lower()

def remove_words(text_data:str,list_of_words_to_remove: typing.List):
    return [item for item in text_data if item not in list_of_words_to_remove]

def collapse_list_to_string(string_list):
    return ' '.join(string_list)

def remove_apostrophes(text):
    text = text.replace("'", "")
    text = text.replace('"', "")
    return text

def text_emotion(df):
    '''
    Takes a DataFrame and a specified column of text and adds 10 columns to the
    DataFrame for each of the 10 emotions in the NRC Emotion Lexicon, with each
    column containing the value of the text in that emotions
    INPUT: DataFrame, string
    OUTPUT: the original DataFrame with ten new columns
    '''
    
    new_df = df.copy()

    emolex_df = pd.read_csv('NRC-Emotion-Lexicon-Wordlevel-v0.92.txt',
                            names=["word", "emotion", "association"],
                            sep='\t')
    
    emolex_words = emolex_df.pivot(index='word',
                                   columns='emotion',
                                   values='association').reset_index()
    
    emotions = emolex_words.columns.drop('word')
    emo_df = pd.DataFrame(0, index=df.index, columns=emotions)

    stemmer = SnowballStemmer("english")
            
    for i, row in new_df.iterrows():
        for word in row['tokenized_sents']:
            word = stemmer.stem(word.lower())
            emo_score = emolex_words[emolex_words.word == word]
            if not emo_score.empty:
                for emotion in list(emotions):
                    emo_df.at[i, emotion] += int(emo_score[emotion])
            
    new_df = pd.concat([new_df, emo_df], axis=1)

    return new_df

def data_processing(directory, df):
    t = TweetTokenizer()
    emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']

    df[conf.datacol] = df[conf.datacol].astype(str).apply(remove_links)
    df['cleaned_text'] = df[conf.datacol].astype(str).apply(style_text)
    df['cleaned_text'] = df['cleaned_text'].astype(str).apply(lambda x: remove_words(x.split(),stopcorpus))
    df['cleaned_text'] = df['cleaned_text'].apply(collapse_list_to_string)
    df['cleaned_text'] = df['cleaned_text'].astype(str).apply(remove_apostrophes)
    df['tokenized_sents'] = df.apply(lambda row: t.tokenize(row['cleaned_text']), axis=1)
    df['word_count'] = df.apply(lambda row: len(row['tokenized_sents']), axis=1)
    df = df[df.word_count  > 0]
    
    df = text_emotion(df)
    
    for emotion in emotions:
        df[emotion] = df[emotion] / df['word_count']
        
    date = datetime.datetime.strptime(df['created_at'].min(), '%Y-%m-%d %H:%M:%S').date()
    
    df.to_pickle(os.path.join(directory, f"{str(date)}.pickle"))
    df.to_excel(os.path.join(directory, f"{str(date)}.xlsx"))

    return

if __name__ == "__main__":
    # Initialise GUI
    conf = parse_args()

    # Read input data
    directory = conf.datadir

    files = []

    for file in os.listdir(directory):
        pkl = os.path.join(directory, file)
        files.append(pickle.load(open(pkl,'rb')))

    # Prepare a directory for translated texts
    sentiment_dir = os.path.join(directory, "sentiment")
    print(f"Output will be found in {sentiment_dir}")

    # Create if does not exist
    if not os.path.exists(sentiment_dir):
        os.mkdir(sentiment_dir)

    # This is only for showing progress bar
    progress_bar_output = io.StringIO()

    with redirect_stderr(progress_bar_output):
        for df in tqdm(files):
            data_processing(sentiment_dir,df)
            print(progress_bar_output.read())

        
        # PyInstaller limitation, cannot implement
        # This creates separate isolated instances of Python and reruns entire file
        # Creating 8 instances of the GUI, not processing each file
        # Parallel(n_jobs=8)(delayed(data_processing)(sentiment_dir, df) for df in tqdm(files))
        # print(progress_bar_output.read())

    meanArr = {"date":[], 'anger':[], 'anticipation':[], 'disgust':[], 'fear':[], 'joy':[], 'negative':[], 'positive':[], 'sadness':[], 'surprise':[], 'trust':[]}

    files = []

    for file in os.listdir(sentiment_dir):
        files.append(pickle.load(open(os.path.join(sentiment_dir, file),'rb')))

    for file in files:
        date = datetime.datetime.strptime(file['created_at'].min(), '%Y-%m-%d %H:%M:%S').date()
        
    meanArr["date"].append(date)
    meanArr["fear"].append(file['fear'].mean()*100)
    meanArr["anger"].append(file['anger'].mean()*100)
    meanArr["joy"].append(file['joy'].mean()*100)
    meanArr["positive"].append(file['positive'].mean())
    meanArr["negative"].append(file['negative'].mean())
    meanArr["anticipation"].append(file['anticipation'].mean())
    meanArr["disgust"].append(file['disgust'].mean())
    meanArr["sadness"].append(file['sadness'].mean())
    meanArr["surprise"].append(file['surprise'].mean())
    meanArr["trust"].append(file['trust'].mean())

    plt.figure(figsize=(20,8))
    plt.title(f'Change in fear, anger and joy for {conf.city_name}', fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Sentiment %', fontsize=15)
    plt.scatter(meanArr["date"], meanArr["fear"], marker = 'x', label='fear', color = '#FF8800')
    plt.scatter(meanArr["date"], meanArr["anger"], marker = 'o', label='anger', color = '#009900')
    plt.scatter(meanArr["date"], meanArr["joy"], marker = '.', label='joy', color = '#0080FF')

    # for emotion in emotions:
    #     plt.scatter(meanArr["date"], meanArr[emotion], marker = 'x')

    def movingaverage(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')

    x_av_fear = movingaverage(meanArr["fear"], 3)
    x_av_joy = movingaverage(meanArr["joy"], 3)
    x_av_anger = movingaverage(meanArr["anger"], 3)
    plt.plot(meanArr["date"], x_av_fear, label='mov. avg. 3D fear', color = '#FF8800')
    plt.plot(meanArr["date"], x_av_joy, label='mov. avg. 3D joy', color = '#0080FF')
    plt.plot(meanArr["date"], x_av_anger, label='mov. avg. 3D anger', color = '#009900')
    plt.legend(loc='upper left')
    plt.savefig(f"{conf.city_name}_sentiment_fear.png")


    # positive / negative sentiment 
    plt.figure(figsize=(20,8))
    plt.title(f'Change in positive and negative sentiment for {conf.city_name}', fontsize=20)
    plt.xlabel('Date', fontsize=15)
    plt.ylabel('Sentiment %', fontsize=15)
    plt.scatter(meanArr["date"], meanArr["positive"], marker = 'x', label='positive', color = '#FF8800')
    plt.scatter(meanArr["date"], meanArr["negative"], marker = 'o', label='negative', color = '#009900')

    # for emotion in emotions:
    #     plt.scatter(meanArr["date"], meanArr[emotion], marker = 'x')

    x_av_positive = movingaverage(meanArr["positive"], 3)
    x_av_negative = movingaverage(meanArr["negative"], 3)
    plt.plot(meanArr["date"], x_av_positive, label='mov. avg. 3D positive', color = '#FF8800')
    plt.plot(meanArr["date"], x_av_negative, label='mov. avg. 3D negative', color = '#0080FF')
    plt.legend(loc='upper left')
    plt.savefig(f"{conf.city_name}_sentiment_posneg.png")


    exclude_words = ["http","https","error","RT", "rt"]

    exclude_words.extend(stopcorpus)

    df = pd.concat(files)

    df['wordcloud'] = df['translated_full_text'].astype(str).apply(remove_links)

    df['wordcloud'] = df['wordcloud'].astype(str).apply(lambda x: x.lower())

    df['wordcloud'] = df['wordcloud'].astype(str).apply(remove_apostrophes)

    df['wordcloud'] = df['wordcloud'].astype(str).apply(lambda x: remove_words(x.split(),exclude_words))

    df['wordcloud'] = df['wordcloud'].apply(lambda x: ' '.join(x))

    # display(output_df['wordcloud'])

    wordcloud = WordCloud(width=1920, height=1080).generate(' '.join(df['wordcloud'].astype(str)))
    wordcloud.to_file(f"{conf.city_name}_wordcloud.png")

    # DONE!
    print(f"Success")