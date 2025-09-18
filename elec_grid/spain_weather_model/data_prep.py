import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# add repo root dir to path lib
import sys
sys.path.append('../../')
# get repo root actual dir (for file reading)
from scripts.system_helpers.repo_root import REPO_ROOT

def load_clean_weather_data():
    # load data~
    df_we = pd.read_csv(f'{REPO_ROOT}/data/spain_weather_model/kagglehub/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather/versions/1/weather_features.csv')
    # for madrid weather
    df_we = df_we[df_we['city_name']=='Madrid']

    # feature cleaning
    df_we = df_we[['dt_iso','temp','humidity','wind_speed','rain_1h','clouds_all','weather_main']]
    # weather_main - clear, cloudy, rainy
    # log wind
    df_we['wind_speed'] = np.log(df_we['wind_speed']).replace([np.inf, -np.inf], 0).fillna(0)
    df_we=df_we.join(pd.get_dummies(df_we['weather_main'].map({'clear':'clear',
                            'clouds':'clouds',
                            'rain':'rain',
                            'mist':'rain',
                            'fog':'rain',
                            'drizzle':'rain',
                            'thunderstorm':'rain',
                            'snow':'rain',
                            'haze':'clouds'}))).drop(['weather_main'],axis=1)
    df_we[['clear',	'clouds',	'rain']] = df_we[['clear',	'clouds',	'rain']].astype(int)

    # date management and filtering
    df_we['dt_iso'] = pd.to_datetime(pd.to_datetime(df_we['dt_iso'],utc=True) + pd.Timedelta(hours=1))
    # create train test flags
    df_we['sample'] = np.where(df_we['dt_iso'].dt.year == 2015, 'ignore', 
                            np.where(df_we['dt_iso'].dt.year == 2016, 'train',
                                        np.where(df_we['dt_iso'].dt.year == 2017, 'train','test')))
    df_we.drop_duplicates(subset = 'dt_iso',inplace=True)
    # remove leap year day
    df_we = df_we[df_we['dt_iso'].dt.date != pd.to_datetime('2016-02-29 00:00:10').date()]
    # check is sorted by date
    df_we = df_we.sort_values('dt_iso')
    df_we.reset_index(inplace=True,drop=True)
    return df_we