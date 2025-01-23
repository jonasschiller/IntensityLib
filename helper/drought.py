import xarray as xr
import reverse_geocoder as rg
from tqdm import tqdm
import pandas as pd
import itertools
import os
import numpy as np
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt

datacenter_loc={"DE":(50.110924,8.682127),"FR":(48.856667,2.351667),"IE":(53.35,-6.26),"IT":(45.4625,9.1864),
                "NL":(52.3702,4.8904),"PL":(52.2167,21.0333),"ES":(40.412,-3.7039),
                "SE":(59.325,18.05),"BE":(50.8466,4.3517),"AT":(48.2083,16.3731)}

def convert_drought_indicator(path='C:\Daten\Foschung\RiskAware\Code\Data\Drought\drought_indicator.nc', save_name="drought_geolocated"):
# Load the NetCDF file
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

    df=df[df['country'].isin(datacenter_loc.keys())]
    df['cdi'] = df.apply(lambda row: dataset.sel(lon=row['lon'], lat=row['lat'])['cdinx'].isel(band=0).values.astype(np.int8), axis=1)
    df.to_csv(save_name)
    return df

def load_drought_indicator(raw_path='C:\Daten\Foschung\RiskAware\Code\Data\Drought\drought_indicator.nc', geo_path='C:\Daten\Foschung\RiskAware\Code\Data\Drought\drought_geolocated.csv'):
    if os.path.exists(geo_path):
        df= pd.read_csv(geo_path,index_col=0)
    else:
        df= convert_drought_indicator(raw_path, geo_path)
    df.reset_index(drop=True, inplace=True)
    df['cdi'] = df['cdi'].apply(lambda x: np.fromstring(x[1:-1], sep=' ',dtype=np.int8))
    df = df[df['cdi'].apply(lambda x: not np.all(x == np.int8(8)))]
    return df

def  calculate_risk(geo_path='C:\Daten\Foschung\RiskAware\Code\Data\Drought\drought_geolocated.csv',p=2,datacenter_loc=datacenter_loc,countries=datacenter_loc.keys()):
    df=load_drought_indicator(geo_path=geo_path)
    if type(countries)==str:
        countries=[countries]
    risk=pd.DataFrame(index=pd.date_range(start='2022-01-01', end='2022-12-31', freq='10D')[:-1], columns=countries)
    for country in countries:
        # Filter for Germany
        df_country = df[df['country'] == country].copy()
        # Build the drought data for Germany
        df_country['cdi'] = df_country['cdi'].apply(lambda x: np.where(x == 6, 1, np.where(x == 5, 2, np.where(x == 4, 3, x))))

        # Calculate distances and weights
        df_country['distance'] = df_country.apply(lambda row: distance.euclidean((row['lat'], row['lon']), (datacenter_loc[country][0],datacenter_loc[country][1])), axis=1)
        df_country['weight'] = np.pow(1 / df_country['distance'],1)
        df_country['weight'] /= df_country['weight'].sum()

        # Calculate weighted drought risk
        risk_country = [np.sum(df_country['cdi'].apply(lambda x: x[i]) * df_country['weight']) for i in range(len(df_country['cdi'].iloc[0]))]

        risk[country] = risk_country

    return risk

def load_weip(countries):
    # Load the data
    weip_22 = pd.read_csv(r"C:\\Daten\\Foschung\\RiskAware\\Data\\Weather\\waterscarcity2022.csv")
    weip = pd.read_csv(r"C:\\Daten\\Foschung\\RiskAware\\Data\\Weather\\waterscarcity.csv")

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


def load_water_weight(countries,p=2):
    risk=calculate_risk(p=2,countries=countries)
    weip=load_weip(countries)
    water_weight=risk[countries]*weip
    water_weight= water_weight.resample('1h').ffill()
    water_weight = water_weight.resample('1h').ffill().reindex(pd.date_range(start=water_weight.index[0], end='2023-01-01 00:00:00', freq='1h'), method='ffill')
    water_weight.index=water_weight.index.tz_localize('UTC')
    return water_weight

