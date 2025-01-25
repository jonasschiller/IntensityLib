import pandas as pd
import numpy as np
import psychrolib as psy
import helper.entsoe_wrapper as entsoe
import helper.flowtracing as ft
import helper.drought as drought
import helper.weather as weather

import os



def get_emission_lca():
    emission_df=pd.read_csv(entsoe.CACHE_DIR+"\\IntFactors\\EmissionLCA.csv",index_col=0,sep=";")
    emission_df = emission_df.replace(',', '.', regex=True).astype(float)
    return emission_df


def get_emission_op():
    emission_df=pd.read_csv(entsoe.CACHE_DIR+"\\IntFactors\\EmissionOp.csv",index_col=0,sep=";")
    emission_df = emission_df.replace(',', '.', regex=True).astype(float)
    return emission_df

def get_water_op():
    water_df=pd.read_csv(entsoe.CACHE_DIR+"\\IntFactors\\WaterConOP.csv",index_col=0,sep=";")
    water_df = water_df.replace(',', '.', regex=True).astype(float)
    #transform to m3/MWh
    water_df=water_df*3.78541
    return water_df

def get_water_lca():
    pass

def get_average_intensity(country: str,start: pd.Timestamp,end: pd.Timestamp,water=True) -> pd.Series:
    if water==True:
        factors=get_water_op()
    else:
        factors=get_emission_op()
    generation = entsoe.get_generation_data_1h(country,start,end)
    generation.drop(columns=[col for col in generation.columns if 'Consumption' in col], inplace=True)
    generation.columns=generation.columns.str.split("_").str[0]
    generation_share = generation.div(generation.sum(axis=1), axis=0)
    emissions=(generation_share*factors[country]).sum(axis=1)
    return emissions

def get_average_intensity_flow(country: str,start: pd.Timestamp,end: pd.Timestamp,water=True) -> pd.Series:
    if water==True:
        factors=get_water_op()
    else:
        factors=get_emission_op()
    generation = ft.get_generation_flowtrace(country,start,end)
    generation.drop(columns=[col for col in generation.columns if 'Consumption' in col], inplace=True)
    generation.columns=generation.columns.str.split("_").str[0]
    generation_share = generation.div(generation.sum(axis=1), axis=0)
    emissions=(generation_share*factors[country]).sum(axis=1)
    return emissions

def get_average_flexible_intensity(country: str,start: pd.Timestamp,end: pd.Timestamp,water=True) -> pd.Series:
    if water==True:
        factors=get_water_op()
    else:
        factors=get_emission_op()
    generation = entsoe.get_generation_data_1h(country,start,end)
    generation.drop(columns=[col for col in generation.columns if 'Consumption' in col], inplace=True)
    generation.columns=generation.columns.str.split("_").str[0]
    generation = generation.drop(columns=generation.columns.intersection([
    'Other renewable', 'Solar', 'Waste', 'Wind Offshore', 'Wind Onshore'
    ]))
    generation_share = generation.div(generation.sum(axis=1), axis=0)
    return (generation_share*factors[country]).sum(axis=1)

def load_direct_WUE(loc,start,end, wCycle = 6):
    '''
    Estimate on-site WUE from wetBulb temperature
    Args:
        weather_path : path for the weather csv data
        wCycle       : number of water cycles
    Return:
        directWue    : estimated on-site water for server cooling
    '''
    
    # 
    wetBulbTemp=weather.get_weather_data(loc, start, end)
    wetBulbTempFahrenheit = wetBulbTemp * 9/5 + 32
    
    directWue = wCycle/(wCycle-1)*(6e-5* wetBulbTempFahrenheit**3 - 0.01 * wetBulbTempFahrenheit**2 + 0.61 * wetBulbTempFahrenheit - 10.4);
    
    # Even though when the temperature is low, we still need 
    # a little bit water for moisture, e.g. 0.05
    return np.clip(directWue, 0.05, None)

def get_complete_WUE(country: str,loc:tuple,start: pd.Timestamp,end: pd.Timestamp,flow=True,weighted=True) -> pd.Series:
    direct=load_direct_WUE(loc,start,end)
    if flow==True:
        indirect=get_average_intensity_flow(country,start,end,water=True)
    else:
        indirect=get_average_intensity(country,start,end,water=True)
    direct.index=direct.index.tz_convert('UTC')
    indirect.index=indirect.index.tz_convert('UTC')
    direct=direct.loc[start:end] 
    indirect=indirect.loc[start:end]
    complete=direct+indirect
    if weighted==True:
        weight=drought.load_water_weight(country,loc,start,end)
        return complete*weight.iloc[:,0]
    return complete

