import pandas as pd
import sys
import numpy as np
sys.path.append('../../')
# get repo root actual dir (for file reading)
from helpers.system_helpers.repo_root import REPO_ROOT
from helpers.data_helpers.elexon_api_helpers import create_dates_list, fetch_physical_data
from helpers.constants import DATA_DIR
from datetime import date
# import xarray as xr
from helpers.data_helpers.ecmwf_helpers import ecmwf_pp

config = {'start_date' : '01/01/2024',
          'end_date' : date.today().strftime('%d/%m/%Y'),
        
        #   'end_date' : '02/01/2024',
          'repd_name' : 'doronell',
          'elexon_id' : ['T_DOREW-1','T_DOREW-2']}

start_date = config['start_date']
end_date = config['end_date']
repd_name = config['repd_name']
elexon_id = config['elexon_id']

# 1. get and save PN data of doronell
dates_list = create_dates_list(start_date,end_date,months_delta=3)
for date1,date2 in dates_list:
    print(date1,date2)
    data_df = fetch_physical_data(date1, 
                        date2, 
                        save_dir = DATA_DIR / 'doronell_data/pn_db/',  
                        cache=True, 
                        unit_ids=['T_DOREW-1','T_DOREW-2'], 
                        multiprocess=True, 
                        pull_data_once = False)

# 2. get and process ECMWF data of doronell location - this data is downloaded manually from https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download
# ECMWF data
# df=ecmwf_pp()