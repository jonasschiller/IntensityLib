import xarray as xr
import reverse_geocoder as rg
import itertools
import numpy as np
import pandas as pd
from scipy.spatial import distance

import requests
import ast
import os
import helper.entsoe_wrapper as entsoe
countries=["AT","PT","ES","FR","IT","GR","ME","BG","RO","RS","HU","SK","SI","CZ","BE","NL","EE","LV","LT","FI","NO","SE","DK","PL","DE","IE","UK"]

def get_drought_indicator(year):
    path=entsoe.CACHE_DIR+'\Drought\drought_'+str(year)+'.nc'
    if not os.path.exists(path):
        download_drought_indicator(year)
        return convert_drought_indicator(year)
    else:
        return convert_drought_indicator(year)

def download_drought_indicator(year):
 # Send out request to CDI
    url="https://drought.emergency.copernicus.eu/services/getData2download?format=nc&scale_id=edo&action=getUrls&year="+str(year)+"&prod_code=cdiad"
    response = requests.get(url)
    download_link = response.text.strip()

    # Extract the URL from the string
    formatted_url = ast.literal_eval(download_link)[0].replace('\\/', '/')

    # Download the file from the extracted link
    file_response = requests.get(formatted_url)
    with open(entsoe.CACHE_DIR+"\\Drought\\drought_"+str(year)+".nc", 'wb') as f:
        f.write(file_response.content)

def convert_drought_indicator(year):
# Load the NetCDF file
    path=entsoe.CACHE_DIR+'\Drought\drought_'+str(year)+'.nc'
    dataset = xr.open_dataset(path)

    # Create a DataFrame with every combination of lon and lat
    combinations = list(itertools.product(dataset.lon.values, dataset.lat.values))
    df = pd.DataFrame(combinations, columns=['lon', 'lat'])

    # Function to get country from coordinates using reverse_geocoder
    def get_country(coords):
        results = rg.search(coords)
        return [result['cc'] for result in results]

    # Prepare coordinates for reverse geocoding
    coords = list(zip(df['lat'], df['lon']))

    # Get countries for all coordinates

    df['country'] = get_country(coords)

    df=df[df['country'].isin(countries)]
    df['cdi'] = df.apply(lambda row: dataset.sel(lon=row['lon'], lat=row['lat'])['cdinx'].isel(band=0).values.astype(np.int8), axis=1)
    df.to_csv(entsoe.CACHE_DIR+'\Drought\drought_geolocated_'+str(year)+'.csv')
    return df

def load_drought_indicator(year):
    path=entsoe.CACHE_DIR+'\Drought\drought_geolocated_'+str(year)+'.csv'
    if os.path.exists(path):
        df= pd.read_csv(path,index_col=0)
    else:
        df= get_drought_indicator(year)
    df.reset_index(drop=True, inplace=True)
    df['cdi'] = df['cdi'].apply(lambda x: np.fromstring(x[1:-1], sep=' ',dtype=np.int8))
    df = df[df['cdi'].apply(lambda x: not np.all(x == np.int8(8)))]
    return df

def calculate_risk(country,loc,start,end,p=2):
    
    risk=None
    for year in range(start.year,end.year+1):
        # Filter for Germany
        df=load_drought_indicator(year)
        df_country = df[df['country'] == country].copy()
        # Build the drought data for Germany
        df_country['cdi'] = df_country['cdi'].apply(lambda x: np.where(x == 6, 1, np.where(x == 5, 2, np.where(x == 4, 3, x))))

        # Calculate distances and weights
        df_country['distance'] = df_country.apply(lambda row: distance.euclidean((row['lat'], row['lon']), (loc[0],loc[1])), axis=1)
        df_country['weight'] = np.pow(1 / df_country['distance'],1)
        df_country['weight'] /= df_country['weight'].sum()

        # Calculate weighted drought risk
        risk_helper = [np.sum(df_country['cdi'].apply(lambda x: x[i]) * df_country['weight']) for i in range(len(df_country['cdi'].iloc[0]))]
        if year==start.year:
            risk=risk_helper
        else:
            risk=np.vstack((risk,risk_helper))
    
    return pd.DataFrame(risk_helper,index=pd.date_range(start=start, end=end, freq='10D')[:-1], columns=[country])


def load_weip(countries):
    # Load the data
    weip_22 = pd.read_csv(entsoe.CACHE_DIR+"\\Weather\\waterscarcity2022.csv")
    weip = pd.read_csv(entsoe.CACHE_DIR+"\\Weather\\waterscarcity.csv")

    # Drop unnecessary columns
    weip_22.drop(columns=['unit_label', 'dimension_label', 'eu_sdg', 'dimension', 'unit', 'geo_label', 'obs_status'], inplace=True)
    weip.drop(columns=['OBS_FLAG', 'unit', 'freq', 'DATAFLOW', 'LAST UPDATE'], inplace=True)

    # Pivot the tables
    weip_table = weip.pivot_table(values='OBS_VALUE', index='TIME_PERIOD', columns='geo')
    weip_table_22 = weip_22.pivot_table(values='obs_value', index='time', columns='geo')

    # Update IT column with common indices
    common_indices = weip_table.index.intersection(weip_table_22.index) 
    weip_table_22.loc[common_indices, 'IT'] = weip_table.loc[common_indices, 'IT']

    # Fill missing values with column mean
    weip_table_22.fillna(weip_table_22.mean(), inplace=True)

    # Calculate the mean for specified labels
    result = weip_table_22.loc[:, countries].mean()
    return result


def load_water_weight(country,loc,start,end,p=2):
    risk=calculate_risk(country,loc,start,end,p=2,)
    weip=load_weip(country)
    water_weight=risk*weip
    water_weight= water_weight.resample('1h').ffill()
    water_weight = water_weight.resample('1h').ffill().reindex(pd.date_range(start=start, end=end, freq='1h'), method='ffill')
    return water_weight

