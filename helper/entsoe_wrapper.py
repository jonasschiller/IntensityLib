# This file serves as a wrapper and cache handler for the entsoe API
# It is based on the entsoe-py library
# All downloaded data is cached in a folder specified by the user
# The cache is checked before downloading data to avoid unnecessary API calls since the ENTSOE-API is very slow

# ToDo
# - Imputation of missing data
# - Handling of outliers

from entsoe import EntsoePandasClient
import pandas as pd
import os
import numpy as np

# Define a global variable for the cache directory
CACHE_DIR = "C:\\Daten\\Foschung\\RiskAware\\Data"
# Entsoe API key
API_KEY="96ebcf8b-a543-4309-b167-322d5e0d5684"
client=EntsoePandasClient(api_key=API_KEY)

def set_cache_dir(cache_dir):
    global CACHE_DIR
    CACHE_DIR = cache_dir
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def get_cache_dir():
    return CACHE_DIR

def set_api_key(api_key):
    global API_KEY
    API_KEY = api_key
    client=EntsoePandasClient(api_key)

def get_api_key():
    return API_KEY

# The functions check whether the data is in the cache or needs to be downloaded
# The data is returned formatted but not cleaned or imputed
# Use one big csv file to store the data, load it and check if time stamps are in the range

def get_generation_data(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    data=None

    #Potentially requires special treatment of countries with changes in accounting
    for year in years:
        filename = os.path.join(CACHE_DIR, "Generation\\Gen_" + country + "_" + str(year) + ".csv")
        # Check if the file exists at all
        if os.path.exists(filename):
            data_helper= pd.read_csv(filename,index_col=0,parse_dates=True)
        else:
            data_helper = client.query_generation(country, start=pd.Timestamp(str(year) + "-01-01",tz="UTC"), end=pd.Timestamp(str(year+1) + "-01-01",tz="UTC"))
            #Fixing data errors present in ENTSOE database
            if country not in ["ME","RS","LV","FI","NO","SE","DK","PL","EE","SI","RO","BG","GR"]:
                data_helper.columns = ['_'.join(col).strip() for col in data_helper.columns.values]
            if country in ["ME","LV","FI"]:
                if "Solar" in data_helper.columns:
                    data_helper["Solar"]=data_helper["Solar"].fillna(0)
            if country == "RS":
                data_helper["Wind Onshore"]=data_helper["Wind Onshore"].fillna(0)
            if country in  ["HU","FR","IT","SK","BE","SE"]:
                data_helper.fillna(0,inplace=True)
            if country == "ES" and year == 2022:
                data_helper["Hydro Pumped Storage_Actual Aggregated"]=data_helper["Hydro Pumped Storage_Actual Aggregated"].fillna(0)
            if country == "IE":
                if "Wind Onshore_Actual Consumption" in data_helper.columns:
                    data_helper.drop(columns=["Wind Onshore_Actual Consumption"],inplace=True)
            
            data_helper.to_csv(filename, index=True)
        if data is None:
            data=data_helper
        else:
            all_columns = data.columns.union(data_helper.columns)
            data = data.reindex(columns=all_columns, fill_value=0)
            data_helper = data_helper.reindex(columns=all_columns, fill_value=0)
            data = pd.concat([data[:-1], data_helper], axis=0)
    data.index = pd.to_datetime(data.index,utc=True)
    return data[start:end]

def get_generation_data_1h(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data=get_generation_data(country, start, end)
    data=data.resample('h').mean()
    return data

def get_generation_data_1h_imp(country: str, start: pd.Timestamp, end: pd.Timestamp,imp_type="ffill") -> pd.DataFrame:
    data=get_generation_data(country, start, end)
    data=data.interpolate(method=imp_type)
    data=data.resample('h').mean()
    data=data.interpolate(method=imp_type)
    return data

def get_generation_data_1h_0(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data=get_generation_data(country, start, end)
    data=data.resample('h').mean()
    data=data.fillna(0)
    return data

def get_generation_dict(countries: list, start: pd.Timestamp, end: pd.Timestamp , imputation_type: str = "no") -> dict:
    """
    Load and process generation data from the cache or the ENTSOE API.

    This function reads generation data and combines them in a dictionary. Three different imputation types are available:
    - "no": No imputation is performed.
    - "imp": Linear interpolation is used to impute missing values.
    - "zero": Missing values are filled with zeros.
    
    Parameters:
    - countryCodes: A list of country codes for which the generation data should be loaded.
    - start: The start date of the data to be loaded.
    - end: The end date of the data to be loaded.
    - imputation_type: The imputation type to be used for missing values.
    Returns:
    - dict of pandas.DataFrame: A dictionary containing the processed generation data for each country code, resampled to hourly frequency.
    - set: A set containing the unique generation types across all countries.
    """

    # Initialize an empty set for generation types
    gen_types = set()
    if imputation_type == "no":
        # Load generation data for each country and update the set of generation types
        gen_dict = {country_code: get_generation_data_1h(country_code, start, end) for country_code in countries}
    elif imputation_type == "imp":
        gen_dict = {country_code: get_generation_data_1h_imp(country_code, start, end) for country_code in countries}
    elif imputation_type == "zero":
        gen_dict = {country_code: get_generation_data_1h_0(country_code, start, end) for country_code in countries}
    for gen_data in gen_dict.values():
        # Drop all columns that contain "Actual Consumption" except for Hydro Pumped Storage
        gen_data.drop(columns=[col for col in gen_data.columns if 'Actual Consumption' in col and 'Hydro Pumped Storage' not in col], inplace=True)
        # Clean up column names
        gen_data.columns = gen_data.columns.str.replace('_Actual Aggregated', '')
        gen_types.update(gen_data.columns)
    # Add all columns that are missing in each country's dataframe
    for country_code in countries:
        gen_dict[country_code] = gen_dict[country_code].reindex(columns=gen_types, fill_value=0)
    return gen_dict, gen_types



def get_load_data(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    data=None
    for year in years:
        filename = os.path.join(CACHE_DIR, "Load\\Load_" + country + "_" + str(year) + ".csv")
        # Check if the file exists at all
        if os.path.exists(filename):
            data_helper= pd.read_csv(filename,index_col=0,parse_dates=True)
        else:
            data_helper = client.query_load(country, start=pd.Timestamp(str(year) + "-01-01",tz="UTC"), end=pd.Timestamp(str(year+1) + "-01-01",tz="UTC"))
            data_helper.columns = ['_'.join(col).strip() for col in data_helper.columns.values]
            data_helper.to_csv(filename, index=True)
        if data is None:
            data=data_helper
        else:
            data=pd.concat([data[:-1],data_helper],axis=0)
    return data[start:end]

def get_load_data_1h(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data=get_load_data(country, start, end)
    data=data.resample('h').mean()
    return data

def get_load_data_1h_imp(country: str, start: pd.Timestamp, end: pd.Timestamp,imp_type="ffill") -> pd.DataFrame:
    data=get_load_data(country, start, end)
    data=data.interpolate(method=imp_type)
    data=data.resample('h').mean()
    data=data.interpolate(method=imp_type)
    return data

def get_load_data_1h_0(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data=get_load_data(country, start, end)
    data=data.resample('h').mean()
    data=data.fillna(0)
    return data

def get_import_data(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    data=None
    for year in years:
        filename = os.path.join(CACHE_DIR, "Import\\Import_" + country + "_" + str(year) + ".csv")
        # Check if the file exists at all
        if os.path.exists(filename):
            data_helper= pd.read_csv(filename,index_col=0,parse_dates=True)
        else:
            data_helper = query_physical_crossborder_allborders(country_code=country, start=pd.Timestamp(str(year) + "-01-01",tz="UTC"),end=pd.Timestamp(str(year+1) + "-01-01",tz="UTC"),export=False, per_hour=True)
            if country =="HU":
                data_helper["SI"]=data_helper["SI"].fillna(0)
            if country == "SK":
                data_helper["UA"]=data_helper["UA"].fillna(0)
            if country == "SI":
                data_helper["HU"]=data_helper["HU"].fillna(0)
            if country == "PL":
                data_helper["SK"]=data_helper["SK"].fillna(0)
            data_helper.to_csv(filename, index=True)
        if data is None:
            data=data_helper
        else:
            #reindex to ensure that all columns are present and filled with zeros
            all_columns = data.columns.union(data_helper.columns)
            data = data.reindex(columns=all_columns, fill_value=0)
            data_helper = data_helper.reindex(columns=all_columns, fill_value=0)
            data=pd.concat([data,data_helper],axis=0)
    return data[start:end]

def get_import_data_1h(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data=get_import_data(country, start, end)
    data=data.resample('h').mean()
    return data

def get_import_data_1h_imp(country: str, start: pd.Timestamp, end: pd.Timestamp,imp_type="ffill") -> pd.DataFrame:
    data=get_import_data(country, start, end)
    data=data.interpolate(method=imp_type)
    data=data.resample('h').mean()
    data=data.interpolate(method=imp_type)
    return data

def get_import_data_1h_0(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data=get_import_data(country, start, end)
    data=data.resample('h').mean()
    data=data.fillna(0)
    return data

def get_import_dict(countries: list, start: pd.Timestamp, end: pd.Timestamp , imputation_type: str = "no") -> dict:
    imp_dict={}
    if imputation_type == "no":
        # Load generation data for each country and update the set of generation types
        imp_dict = {country_code: get_import_data_1h(country_code, start, end) for country_code in countries}
    elif imputation_type == "imp":
        imp_dict = {country_code: get_import_data_1h_imp(country_code, start, end) for country_code in countries}
    elif imputation_type == "zero":
        imp_dict = {country_code: get_import_data_1h_0(country_code, start, end) for country_code in countries}
    return imp_dict

def get_export_data(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    data=None
    for year in years:
        filename = os.path.join(CACHE_DIR, "Export\\Export_" + country + "_" + str(year) + ".csv")
        # Check if the file exists at all
        if os.path.exists(filename):
            data_helper= pd.read_csv(filename,index_col=0,parse_dates=True)
        else:
            data_helper = query_physical_crossborder_allborders(country, end=pd.Timestamp(str(year+1) + "-01-01",tz="UTC"), start=pd.Timestamp(str(year) + "-01-01",tz="UTC"),export=True)
            if country =="HU":
                data_helper["SI"]=data_helper["SI"].fillna(0)
            if country == "SK":
                data_helper["UA"]=data_helper["UA"].fillna(0)
            if country == "SI":
                data_helper["HU"]=data_helper["HU"].fillna(0)
            if country == "PL":
                data_helper["SK"]=data_helper["SK"].fillna(0)
            data_helper.to_csv(filename, index=True)
        if data is None:
            data=data_helper
        else:
            all_columns = data.columns.union(data_helper.columns)
            data = data.reindex(columns=all_columns, fill_value=0)
            data_helper = data_helper.reindex(columns=all_columns, fill_value=0)
            data=pd.concat([data[:-1],data_helper],axis=0)
    return data[start:end]

def get_export_data_1h(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data=get_export_data(country, start, end)
    data=data.resample('h').mean()
    return data

def get_export_data_1h_imp(country: str, start: pd.Timestamp, end: pd.Timestamp,imp_type="ffill") -> pd.DataFrame:
    data=get_export_data(country, start, end)
    data=data.interpolate(method=imp_type)
    data=data.resample('h').mean()
    data=data.interpolate(method=imp_type)
    return data

def get_export_data_1h_0(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    data=get_export_data(country, start, end)
    data=data.resample('h').mean()
    data=data.fillna(0)
    return data

def get_export_dict(countries: list, start: pd.Timestamp, end: pd.Timestamp , imputation_type: str = "no") -> dict:
    exp_dict={}
    if imputation_type == "no":
        # Load generation data for each country and update the set of generation types
        exp_dict = {country_code: get_export_data_1h(country_code, start, end) for country_code in countries}
    elif imputation_type == "imp":
        exp_dict = {country_code: get_export_data_1h_imp(country_code, start, end) for country_code in countries}
    elif imputation_type == "zero":
        exp_dict = {country_code: get_export_data_1h_0(country_code, start, end) for country_code in countries}
    return exp_dict

def get_installed_capacity(country: str,start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    data=None
    for year in years:
        filename = os.path.join(CACHE_DIR, "Capacity\\Capacity_" + country + "_" + str(year) + ".csv")
        # Check if the file exists at all
        if os.path.exists(filename):
            data_helper= pd.read_csv(filename,index_col=0)
        else:
            data_helper = client.query_installed_generation_capacity(country, start=pd.Timestamp(str(year) + "-01-01",tz="UTC"), end=pd.Timestamp(str(year+1) + "-01-01",tz="UTC"))
            data_helper.index=data_helper.index.year
            data_helper.to_csv(filename, index=True)
        if data is None:
            data=data_helper
        else:
            data=pd.concat([data,data_helper],axis=0)
    return data.loc[start.year:end.year]

def get_outages(country: str,start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    data=None
    for year in years:
        filename = os.path.join(CACHE_DIR, "Outages\\Outages_" + country + "_" + str(year) + ".csv")
        # Check if the file exists at all
        if os.path.exists(filename):
            data_helper= pd.read_csv(filename,index_col=0)
            data_helper['start'] = pd.to_datetime(data_helper['start'],utc=True)
            data_helper['end'] = pd.to_datetime(data_helper['end'],utc=True)
        else:
            data_helper = client.query_unavailability_of_generation_units(country, start=pd.Timestamp(str(year) + "-01-01",tz="UTC"), end=pd.Timestamp(str(year) + "-12-31",tz="UTC"))
            # Drop unnecessary columns
            columns_to_drop = ["curvetype", "biddingzone_domain", "mrid", "production_resource_location", "production_resource_name", "production_resource_psr_name", "pstn", "qty_uom", "revision"]
            data_helper.drop(columns=columns_to_drop, inplace=True)

            # Reset index and drop cancelled outages
            data_helper.reset_index(drop=True, inplace=True)
            data_helper = data_helper[data_helper['docstatus'] != 'Cancelled']
            # Ensure 'start' column is in datetime format and round to hourly resolution
            data_helper['start'] = pd.to_datetime(data_helper['start']).dt.round('h', ambiguous='NaT',nonexistent='shift_forward')
            data_helper['start'] = data_helper['start'].dt.tz_convert('UTC')
            data_helper['end'] = pd.to_datetime(data_helper['end']).dt.round('h', ambiguous='NaT',nonexistent='shift_forward')
            data_helper['end'] = data_helper['end'].dt.tz_convert('UTC')
            # Convert 'avail_qty' column to float
            data_helper['avail_qty'] = data_helper['avail_qty'].astype(float)
            data_helper.to_csv(filename, index=True)
        if data is None:
            data=data_helper
        else:
            data=pd.concat([data,data_helper],axis=0)
    return data[(data['start'] <= end) & (data['end'] >= start)].drop_duplicates()


def get_missing_capacity(country: str,start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    outages = get_outages(country, start, end)
    # Create a DataFrame to store the missing capacity for each plant type
    missing_capacity = pd.DataFrame(columns=outages.plant_type.unique(), index=pd.date_range(start, end, freq='h'))

    # Pre-compute the filters for each plant type
    for plant_type in outages.plant_type.unique():
        filtered_outages = outages[outages["plant_type"] == plant_type]
        for current_time in pd.date_range(start, end, freq='h'):
            current_outages = filtered_outages[(filtered_outages['start'] <= current_time) & (filtered_outages['end'] >= current_time)]
            missing_capacity.loc[current_time, plant_type] = current_outages.nominal_power.sum() - current_outages.avail_qty.sum()

    return missing_capacity

def get_missing_capacity_percentage(country: str,start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    outages=get_outages(country,start,end)
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    # Create a DataFrame to store the missing capacity for each plant type
    missing_capacity = get_missing_capacity(country,start,end)
    installed_capacity = get_installed_capacity(country,start,end)

    # Create a DataFrame to store the percentage of missing capacity
    missing_capacity_percentage = pd.DataFrame(columns=outages.plant_type.unique(), index=pd.date_range(start, end, freq='h'))

    # Calculate the percentage of missing capacity for each plant type
    for current_time in pd.date_range(start, end, freq='h'):
        for plant_type in missing_capacity.columns:
            missing_capacity_percentage[plant_type] = missing_capacity[plant_type].astype(float) / installed_capacity[year,plant_type]
    return missing_capacity_percentage

def get_price_data(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    data=None
    if country== "DE":
        country="DE_LU"
    elif country=="IT":
        country="IT_CNOR"
    elif country=="SE":
        country="SE_3"
    elif country=="IE":
        country="IE_SEM"
    for year in years:
        filename = os.path.join(CACHE_DIR, "Price\\Price_" + country + "_" + str(year) + ".csv")
        # Check if the file exists at all
        if os.path.exists(filename):
            data_helper= pd.read_csv(filename,index_col=0,parse_dates=True)
            data_helper.index=pd.to_datetime(data_helper.index,utc=True)
        else:
            data_helper = client.query_day_ahead_prices(country, start=pd.Timestamp(str(year) + "-01-01",tz="UTC"), end=pd.Timestamp(str(year+1) + "-01-01",tz="UTC"))
            data_helper.index=data_helper.index.tz_convert('UTC')
            data_helper.to_csv(filename, index=True)
        if data is None:
            data=data_helper
        else:
            data=pd.concat([data.iloc[:-1],data_helper],axis=0)
        # check if timesteps are missing and fill with data from the previous day
    if country=="IE_SEM":
        date_range = pd.date_range(start, end, freq='1h')
        missing_dates = date_range.difference(data.index)
        data_frame=pd.DataFrame(data,index=date_range,columns=data.columns)
        for missing_date in missing_dates:
            data_frame.loc[missing_date] = data_frame.loc[missing_date - pd.Timedelta(hours=48)]
        return data_frame[start:end]
    return pd.DataFrame(data[start:end])

# Helper function for non-included countries where API does not have the correct neighbours
countryCodes=["AT","PT","ES","FR","IT","GR","ME","BG","RO","RS","HU","SK","SI","CZ","BE","NL","EE","LV","LT","FI","NO","SE","DK","PL","DE","IE","UK"]

NEIGHBOURS = {
    "AT": ["CZ","DE","HU","IT","SI","CH"],
    "PT": ["ES"],
    "ES": ["PT","FR"],
    "FR": ["ES","IT","CH","BE","DE","UK"],
    "IT": ["FR","SI","CH","AT","GR","MT","ME"],
    "GR": ["AL","BG","TR","IT","MK"],
    "ME": ["AL","BA","IT","XK","RS"],
    "BG": ["GR","RO","RS","TR","MK"],
    "RO": ["BG","HU","MD","RS","UA"],
    "RS": ["AL","BA","BG","HR","HU","XK","ME","MK","RO"],
    "HU": ["AT","HR","RO","RS","SK","SI","UA"],
    "SK": ["CZ","HU","PL","UA"],
    "SI": ["AT","HR","IT","HU"],
    "CZ": ["AT","DE","PL","SK"],
    "BE": ["FR","DE","LU","NL","UK"],
    "NL": ["BE","DE","UK","NO","DK"],
    "EE": ["FI","LV","RU"],
    "LV": ["EE","LT","RU"],
    "LT": ["BY","LV","PL","RU","SE"],
    "FI": ["EE","RU","SE","NO"],
    "NO": ["DK","FI","DE", "SE", "NL","UK"],
    "SE": ["NO", "FI", "DK", "DE", "PL","LT"],
    "DK": ["DE","SE","NO","UK","NL"],
    "PL": ["DE","LT","CZ","SK","UA","SE"],
    "DE": ["AT","BE","CZ","DK","FR", "LU", "NL", "NO","PL", "SE","CH" ],
    "UK": ["IE", "FR", "NL", "BE","NO","DK"],
    "IE": ["UK"],
}

def query_physical_crossborder_allborders(country_code, start: pd.Timestamp,
                     end: pd.Timestamp, export: bool, per_hour: bool = False) -> pd.DataFrame:
        """
        Adds together all physical cross-border flows to a country for a given direction
        The neighbours of a country are given by the NEIGHBOURS mapping

        if export is True then all export flows are returned, if False then all import flows are returned
        some borders have more then once data points per hour. Set per_hour=True if you always want hourly data,
        it will then thake the mean
        """
        imports = []
        for neighbour in NEIGHBOURS[country_code]:
            try:
                if export:
                    im = client.query_crossborder_flows(country_code_from=country_code,
                                                      country_code_to=neighbour,
                                                      end=end,
                                                      start=start,
                                                      lookup_bzones=True)
                else:
                    im = client.query_crossborder_flows(country_code_from=neighbour,
                                                      country_code_to=country_code,
                                                      end=end,
                                                      start=start,
                                                      lookup_bzones=True)
            except:
                print("Could not load data for ", neighbour)
                continue
            im.name = neighbour
            imports.append(im)
            im.index = im.index.tz_convert('UTC')
          
        df = pd.concat(imports, axis=1, sort=True)
        df.index = df.index.tz_convert('UTC')
        # drop columns that contain only zero's
        df = df.loc[:, (df != 0).any(axis=0)]
        # Typically converts Timezone -> need to watch out for proper alignment
        df = df.truncate(before=start, after=end)
        df['sum'] = df.sum(axis=1)
        if per_hour:
            df = df.resample('h').mean()

        return df