#libraries
import pickle
import pandas as pd
from datetime import datetime, timezone

#load pickle file for sentiment data
city_sent = pickle.load(open("D:\SDSU BDA\HDMA\Italian Regions\Milan1.pickle", "rb"))

#print(city_sent.head())

city_sent['DateTime'] = pd.to_datetime(city_sent['created_at'])

#city_sent['DateTime'].dtypes
#city_sent.dtypes
#
#city_sent_dt=city_sent['DateTime'].strftime("%Y")

city_sent = city_sent.set_index(['DateTime'])
anger = pd.DataFrame(data=city_sent['anger'].resample('D').mean())
joy = pd.DataFrame(data=city_sent['joy'].resample('D').mean()) 
fear = pd.DataFrame(data=city_sent['fear'].resample('D').mean())

final = anger.join(joy)
final = final.join(fear)
final.columns = ['Anger', 'Joy', 'Fear']

print(final.head())

final.to_csv(r'milan1_sent.csv', index = False, header=True)

#save sentiment dataframe to a csv
#city_sent_all.to_csv(r'rome_sent.csv', index = False, header=True)
#reading the covid data in
#city_cases = pd.read_csv("covid_data_palermo.csv")
#check and see if loaded correctly
#city_cases.head()

#pull out needed columns and convert to a dataframe
#city_cases_all = pd.DataFrame(city_cases, columns = ['Date', 'Cumulative Positive', 'New #positives', 'Deceased', 'New dead', 'Currently positive', 'Currently hospitalized', 'Hospitalized with symptoms', 'Intensive care', 'Home isolation', 'Dischagred healed', 'Swabs'])

#city_cases_all=city_cases_all.fillna(0)
#city_cases_all.head()

#city_cases_all.rename(columns = {'Date':'DateTime'}, inplace = True)
#city_cases_all.head()