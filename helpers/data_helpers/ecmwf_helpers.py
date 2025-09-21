import numpy as np
import pandas as pd
import xarray as xr
import sys
sys.path.append('../../')
# get repo root actual dir (for file reading)
from helpers.system_helpers.repo_root import REPO_ROOT

def linear_approximation(df, col):

    windfarm_lat = 57.33
    windfarm_long = -3.13
    lat1 = np.ceil(windfarm_lat/0.25)  * 0.25
    lat2 = np.floor(windfarm_lat/0.25)  * 0.25
    long1 = np.ceil(windfarm_long/0.25)  * 0.25
    long2 = np.floor(windfarm_long/0.25)  * 0.25

    assert np.abs(lat2-lat1) == np.abs(long1-long2) == 0.25, print(np.abs(lat2-lat1), np.abs(long1-long2))
    # choose the four values the box within which is our wind farm
    df = df[(df['latitude'].isin([lat1,lat2])) & (df['longitude'].isin([long1,long2]))]
    # sort by lat and long so we ensure that we take the diff of the smallest to biggest
    df = df[['time','latitude','longitude',col]].sort_values(['time','latitude','longitude'])

    # get the gradients of change in wind speed by change in coord withing the quadrant
    df = df.join(df.groupby(['time','latitude'])[[col]].diff().rename(columns = {col:'mlong'})/-0.25)
    df = df.join(df.groupby(['time','longitude'])[[col]].diff().rename(columns = {col:'mlat'})/-0.25)
    # and align those to the row where both at ad long has changed (the patial derivatives, ( we have assumed dy/dx and dy/dz are independent here)
    df = df.drop(['mlong','mlat'], axis=1).join(df.groupby('time')[['mlong','mlat']].transform(lambda x: x.dropna().iloc[0] if x.notna().any() else pd.NA))

    # get the wind speed in the moiddle of the quadrant going from each of the two sides and take the average
    df['latitude_wf'] = windfarm_lat
    df['longitude_wf'] = windfarm_long
    df['latitude_wf_diff'] = df['latitude_wf'] - df['latitude']
    df['longitude_wf_diff'] = df['longitude_wf'] - df['longitude']
    df[f'{col}_wf'] = df[col] + df['mlong'] * df['longitude_wf_diff'] + df['mlat'] * df['latitude_wf_diff']
    df = df.join(df.groupby(['time'])[[f'{col}_wf']].transform(lambda x: (x.iloc[0] + x.iloc[3])/2).rename(columns = {f'{col}_wf':f'{col}_wf_avg'}))
    # just keep the info we need now
    df = df[['time',f'{col}_wf_avg']].rename(columns = {f'{col}_wf_avg':col}).drop_duplicates()

    df[col] = pd.to_numeric(df[col],errors='coerce')
    df[col] = pd.to_numeric(df[col],errors='coerce')
    return df

def get_wind_angle(df_gcmwf):
    df_gcmwf['angle'] = np.arctan(np.abs(df_gcmwf['v100'])/np.abs(df_gcmwf['u100']))*180/np.pi
    # df_gcmwf['angle'] = np.arcsin(df_gcmwf['v100']/df_gcmwf['u100'])*180/np.pi
    df_gcmwf['angle'] = np.where((df_gcmwf['v100'] > 0) & (df_gcmwf['u100'] > 0), df_gcmwf['angle'], np.where(
            (df_gcmwf['v100'] < 0) & (df_gcmwf['u100'] > 0), 360-df_gcmwf['angle'], np.where(
                    (df_gcmwf['v100'] > 0) & (df_gcmwf['u100'] < 0), 180-df_gcmwf['angle'], np.where(
                        (df_gcmwf['v100'] < 0) & (df_gcmwf['u100'] < 0), 270-df_gcmwf['angle'], df_gcmwf['angle']))))
    return df_gcmwf

def ecmwf_pp():
    
    xr_raw = xr.open_dataset(f'{REPO_ROOT}/data/doronell_data/doronell_ecmwf.grib', engine="cfgrib")
    df_v100 = xr_raw['v100'].to_dataframe().reset_index()
    df_u100 = xr_raw['u100'].to_dataframe().reset_index()

    # get point of wind at wind farm location
    df_gcmwf = linear_approximation(df_v100,'v100')
    df_gcmwf = df_gcmwf.merge(linear_approximation(df_u100,'u100'), on = ['time'])
    # get actual wind speed using pythag
    df_gcmwf['h100'] = np.sqrt(df_gcmwf['v100']**2 + df_gcmwf['u100']**2)
    # get wind direction using pythag
    df_gcmwf = get_wind_angle(df_gcmwf)


    min_time_str = df_gcmwf.time.min().strftime(format = '%H-%d/%m/%Y')
    max_time_str = df_gcmwf.time.max().strftime(format = '%H-%d/%m/%Y')

    df_gcmwf.to_feather(f'{REPO_ROOT}/data/doronell_dataweather/{min_time_str}_{max_time_str}_ws.feather')
    return df_gcmwf