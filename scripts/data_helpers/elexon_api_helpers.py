# import concurrent.futures
import logging
# import os
# import sqlite3
import requests
# import time
# from pathlib import Path

# import numpy as np
import pandas as pd
# from sqlalchemy import create_engine
# from sqlalchemy.exc import OperationalError, IntegrityError
from sp2ts import dt2sp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def add_utc_timezone(datetime):
    """ Add utc timezone to datetime. """
    if datetime.tzinfo is None:
        datetime = datetime.tz_localize('UTC')
    else:
        datetime = datetime.tz_convert('UTC')
    return datetime


def call_physbm_api(start_date, end_date, unit=None):
    """Thin wrapper to allow kwarg passing with starmap"""
    logger.info(f"Calling BOAS API for {unit}")

    # Nedd to call PNs and BOALs separately in new API

    # "https://data.elexon.co.uk/bmrs/api/v1/balancing/physical/all?dataset={dataset}&settlementDate={settlementDate}&settlementPeriod={settlementPeriod}&format=json"
    datetimes = pd.date_range(start_date, end_date, freq="30min")
    data_df = []
    for datetime in datetimes:
        logger.info(f"Getting PN from {datetime}")

        datetime = add_utc_timezone(datetime)

        date, sp = dt2sp(datetime)
        url = f"https://data.elexon.co.uk/bmrs/api/v1/balancing/physical/all?dataset=PN&settlementDate={date}&settlementPeriod={sp}"
        if unit is not None:
            url = url + f"&bmUnit={unit}"
        url = url + "&format=json"

        r = requests.get(url)

        data_one_settlement_period_df = pd.DataFrame(r.json()["data"])
        data_df.append(data_one_settlement_period_df)

    data_pn_df = pd.concat(data_df)

    datetimes = pd.date_range(start_date, end_date, freq="30min")
    data_df = []
    for datetime in datetimes:
        logger.info(f"Getting BOALF from {datetime}")
        boalf_end_datetime = (datetime + pd.Timedelta(minutes=30)).tz_localize(None)
        boalf_start_datetime = (datetime - pd.Timedelta(minutes=30)).tz_localize(None)
        url = f"https://data.elexon.co.uk/bmrs/api/v1/datasets/BOALF?from={boalf_start_datetime}&to={boalf_end_datetime}"
        if unit is not None:
            url = url + f"&bmUnit={unit}"
        url = url + "&format=json"

        r = requests.get(url)

        data_one_settlement_period_df = pd.DataFrame(r.json()["data"])
        data_df.append(data_one_settlement_period_df)

    data_boa_df = pd.concat(data_df)

    # rename bmUnit to bmUnitID
    data_pn_df.rename(columns={"bmUnit": "bmUnitID"}, inplace=True)
    data_boa_df.rename(columns={"bmUnit": "bmUnitID"}, inplace=True)

    # drop dataset column
    data_boa_df.drop(columns=["nationalGridBmUnit"], inplace=True)
    data_pn_df.drop(columns=["nationalGridBmUnit"], inplace=True)
    data_boa_df.drop(columns=["settlementPeriodTo"], inplace=True)
    data_boa_df.drop(columns=["amendmentFlag"], inplace=True)
    data_boa_df.drop(columns=["storFlag"], inplace=True)

    # rename LevelFrom to bidOfferLevelFrom
    data_pn_df.rename(columns={"dataset": "recordType"}, inplace=True)
    data_boa_df.rename(columns={"dataset": "recordType"}, inplace=True)
    data_boa_df.rename(columns={"acceptanceNumber": "Accept ID"}, inplace=True)
    data_boa_df.rename(columns={"settlementPeriodFrom": "settlementPeriod"}, inplace=True)
    data_boa_df.rename(columns={"deemedBoFlag": "deemedBidOfferFlag"}, inplace=True)
    data_boa_df.rename(columns={"rrFlag": "rrScheduleFlag"}, inplace=True)

    data_df = pd.concat([data_boa_df, data_pn_df], axis=0)
    data_df['local_datetime'] = pd.to_datetime(data_df['timeFrom'])

    # remove anything after end_date
    data_df = data_df[data_df['local_datetime'] <= end_date]

    return data_df

# specific wind units only - bug fix with retries and also add aprallel api calls
# df_bm_units = pd.read_csv('BMU.csv')
# wind_units = df_bm_units[df_bm_units["FUEL TYPE"] == "WIND"]["SETT_BMU_ID"].unique()
# data_df=pd.DataFrame()
# for unit in wind_units:
#     data_df = pd.concat([data_df, call_physbm_api(start_date, end_date, unit)])




