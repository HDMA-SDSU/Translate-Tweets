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
import numpy as np
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

    # DONE!
    print(f"Success")