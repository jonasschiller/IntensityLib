import pandas as pd
import numpy as np
import psychrolib as psy

import os

psy.SetUnitSystem(psy.SI)

def get_emission_lca():
    emission_df=pd.read_csv("C:\\Daten\\Foschung\\RiskAware\\Data\\IntFactors\\EmissionLCA.csv",index_col=0,sep=";")
    emission_df = emission_df.replace(',', '.', regex=True).astype(float)
    return emission_df


def get_emission_op():
    emission_df=pd.read_csv("C:\\Daten\\Foschung\\RiskAware\\Data\\IntFactors\\EmissionOp.csv",index_col=0,sep=";")
    emission_df = emission_df.replace(',', '.', regex=True).astype(float)
    return emission_df

def get_water_op():
    water_df=pd.read_csv("C:\\Daten\\Foschung\\RiskAware\\Data\\IntFactors\\WaterConOP.csv",index_col=0,sep=";")
    water_df = water_df.replace(',', '.', regex=True).astype(float)
    #transform to m3/MWh
    water_df=water_df*3.78541
    return water_df

def get_water_lca():
    pass


def get_average_intensity(data, factors,country):
    data.drop(columns=[col for col in data.columns if 'Consumption' in col], inplace=True)
    data.columns=data.columns.str.split("_").str[0]
    data_share = data.div(data.sum(axis=1), axis=0)
    return (data_share*factors[country]).sum(axis=1)

def load_weather_data(country):
    data=pd.read_csv("Data\Weather\Wea_"+country+"_2022.csv",index_col=1)
    data=data.loc[:,["tmpf","relh","alti"]]
    data.replace("M",np.nan,inplace=True)
    data=data.astype(float)
    #Interpolate missing values as mean of the previous and next value
    data=data.interpolate(method="linear",axis=0)
     # Check for NaNs after interpolation
    print("After interpolation:\n", data.isna().sum())
    data.loc[:,"alti"]=data.loc[:,"alti"]*33.8639*100
    data.loc[:,"tmpc"]=(data.loc[:,"tmpf"]-32)*5/9
    data.loc[:,"relh"]=(data.loc[:,"relh"])/100
    data.drop(columns="tmpf",inplace=True)
    data.index=pd.to_datetime(data.index)
    data=data.resample("h").mean()
    data.interpolate(method="linear",axis=0,inplace=True)
    return data

def getWetBulpTemperature(temp_data):
    twet_bulb_series = temp_data.apply(lambda row: psy.GetTWetBulbFromRelHum(row['tmpc'], row['relh'], row['alti']), axis=1)
    return twet_bulb_series

def load_direct_WUE(country, wCycle = 6):
    '''
    Estimate on-site WUE from wetBulb temperature
    Args:
        weather_path : path for the weather csv data
        wCycle       : number of water cycles
    Return:
        directWue    : estimated on-site water for server cooling
    '''
    
    # 
    temp_data=load_weather_data(country)
    wetBulbTemp=getWetBulpTemperature(temp_data)
    
    directWue = wCycle/(wCycle-1)*(6e-5* wetBulbTemp**3 - 0.01 * wetBulbTemp**2 + 0.61 * wetBulbTemp - 10.4);
    
    # Even though when the temperature is low, we still need 
    # a little bit water for moisture, e.g. 0.05
    return np.clip(directWue, 0.05, None)