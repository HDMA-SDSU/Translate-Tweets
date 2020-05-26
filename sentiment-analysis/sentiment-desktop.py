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

stopcorpus: typing.List = stopwords.words('english')

@Gooey(program_name="COVID-Crowdfight Sentiment Analysis", progress_regex=r"(\d+)%")
def parse_args():
    prog_descrip = "Apply sentiment analyis"

    parser = GooeyParser(description=prog_descrip)

    parser.add_argument(
        "datadir",
        metavar="Translated Twitter Data",
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
        "--parallel_en",
        metavar="Enable Parallel compute",
        help="Uses multicore processing",
        action = "store_true",
        default=True,
        widget="CheckBox",
    )

    parser.add_argument(
        "--save_pickle",
        metavar="Save files as pickles",
        help="Uses raw pandas dataframe format (faster)",
        action = "store_true",
        default=True,
        widget="CheckBox",
    )

    parser.add_argument(
        "--save_excel",
        metavar="Save files as Excel",
        help="Uses Microsoft Excel file format",
        action = "store_true",
        default=False,
        widget="CheckBox",
    )

    parser.add_argument(
        "--cpu_cores",
        metavar="Number of Cores to use",
        help="If multicore processing is used, python will spawn multiple instances",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--datacol",
        metavar="Data Column Name",
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

    exclude_words = ["http","https","error","RT", "rt"]
    exclude_words.extend(stopcorpus)

    df['cleaned_text'] = df[conf.datacol].astype(str).apply(remove_links)
    df['cleaned_text'] = df['cleaned_text'].astype(str).apply(style_text)
    df['cleaned_text'] = df['cleaned_text'].astype(str).apply(lambda x: remove_words(x.split(),exclude_words))
    df['cleaned_text'] = df['cleaned_text'].apply(collapse_list_to_string)
    df['cleaned_text'] = df['cleaned_text'].astype(str).apply(remove_apostrophes)
    df['tokenized_sents'] = df.apply(lambda row: t.tokenize(row['cleaned_text']), axis=1)
    df['word_count'] = df.apply(lambda row: len(row['tokenized_sents']), axis=1)
    df = df[df.word_count  > 0]
    
    df = text_emotion(df)
    
    for emotion in emotions:
        df[emotion] = df[emotion] / df['word_count']
        
    date = datetime.datetime.strptime(df['created_at'].min(), '%Y-%m-%d %H:%M:%S').date()
    
    if conf.save_pickle:
        df.to_pickle(os.path.join(directory, f"{str(date)}.pickle"))
    if conf.save_excel:
        df.to_excel(os.path.join(directory, f"{str(date)}.xlsx"))

    print(progress_bar_output.read())
    return

if __name__ == "__main__":
    # Initialise GUI
    conf = parse_args()

    # Read input data
    directory = conf.datadir

    files = []

    for file in os.listdir(directory):
        filename = os.path.join(directory, file)
        if filename.endswith(".pickle")  or filename.endswith(".pkl"):
            df = pickle.load(open(filename,'rb'))
            files.append(df)
        elif filename.endswith(".xlsx"):
            df = pd.read_excel(filename)
            files.append(df)
        else:
            raise NotImplementedError("File type not supported")

    # Prepare a directory for translated texts
    sentiment_dir = os.path.join(directory, "sentiment")
    print(f"Output will be found in {sentiment_dir}")

    # Create if does not exist
    if not os.path.exists(sentiment_dir):
        os.mkdir(sentiment_dir)

    # This is only for showing progress bar
    progress_bar_output = io.StringIO()

    with redirect_stderr(progress_bar_output):
        if(not conf.parallel_en):
            for df in tqdm(files, file=sys.stdout):
                data_processing(sentiment_dir,df)
        else:
            Parallel(n_jobs=conf.cpu_cores)(delayed(data_processing)(sentiment_dir, df) for df in tqdm(files, file=sys.stdout))

    # DONE!
    print(f"Success")