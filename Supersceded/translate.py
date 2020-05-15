import requests 
import json
import numpy as np
import pandas as pd

# DeepL API key
api_key = "insert-key-here"

# Twitter dataset
df = pd.read_csv('insert_origninal_data_here.csv')

# Add new coloumn for transations
df.insert(7, "translated_full_text", "", True)

# Create empty list
my_list = []

# Translate all tweets and add to list
# "text" parameter can be an array however there is a limit which is not defined in the documentation
for tweet in df["full_text"]:
    parameters = {"text": tweet, "source_lang": "IT", "target_lang": "EN", "auth_key": api_key}
    response = requests.get("https://api.deepl.com/v2/translate", params=parameters)
    data = response.json()
    for item in data.values():
        for key in item:
            my_list.append(key['text'])

# Copy list into data frame
df['translated_full_text'] = my_list

# Pickle dataset
df.to_pickle("city.pkl")