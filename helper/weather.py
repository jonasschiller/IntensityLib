import pandas as pd
import numpy as np
import psychrolib as psy
import helper.entsoe_wrapper as entsoe
import os
from datetime import date, timedelta
import httpx
import warnings
psy.SetUnitSystem(psy.SI)


def get_weather_data(loc, start, end):
    start_year = start.year
    end_year = end.year
    airport=get_closest_weather_station(loc).iloc[0]
    for year in range(start_year, end_year+1):
        data_helper = load_weather_data(airport, year)
        data_helper = getWetBulpTemperature(data_helper)
        if year == start_year:
            data = data_helper
        else:
            data = pd.concat([data, data_helper], axis=0)
    try:
        data.index = data.index.tz_localize("UTC")
    except TypeError:
        data.index = data.index.tz_convert("UTC")
    data=data.resample("h").interpolate(method="linear")
    return data[start:end]

def fetch_weather_data(station_id,year):
    """Download data we are interested in!"""
    localfn = f"Data/Airports/Wea_{station_id}_{year}.csv"
    if os.path.isfile(localfn):
        print(f"- Cowardly refusing to over-write existing file: {localfn}")
        return
    print(f"+ Downloading for {station_id}")
    startdt=date(year,1,1)
    enddt = date(year+1,1,1)
    uri = (
        "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"
        f"station={station_id}&year1={startdt.year}&month1={startdt.month}&day1={startdt.day}&"
        f"year2={enddt.year}&month2={enddt.month}&day2={enddt.day}&"
        "tz=Etc%2FUTC&format=onlycomma&latlon=no&elev=no&missing=M&trace=T&"
        "direct=yes&report_type=3"
    )
    resp = httpx.get(uri, timeout=300)
    with open(localfn, "w", encoding="utf-8") as fh:
        fh.write(resp.text)

def load_weather_data(loc_code, year):
    path=entsoe.CACHE_DIR+f"\\Airports\\Wea_{loc_code}_{year}.csv"
    if not os.path.exists(path):
        fetch_weather_data(loc_code, year)
    pd.set_option('future.no_silent_downcasting', True)
    data = pd.read_csv(path, index_col=1).replace("M", np.nan)
    data = data[["tmpf", "relh", "alti"]].astype(float).interpolate(method="linear")
    data["alti"] *= 33.8639 * 100
    data["tmpc"] = (data["tmpf"] - 32) * 5 / 9
    data["relh"] /= 100
    data.drop(columns="tmpf", inplace=True)
    data.index = pd.to_datetime(data.index)
    data = data.resample("h").mean().interpolate(method="linear")
    return data

def get_closest_weather_station(loc):
    airport = pd.read_csv(entsoe.CACHE_DIR+"\\Airports\Airports.csv",sep=";",index_col=0,dtype={"Latitude":float,"Longitude":float})
    dist=abs(airport['Latitude']-loc[0])+abs(airport['Longitude']-loc[1])
    return airport.iloc[dist.idxmin()]

def getWetBulpTemperature(temp_data):
    twet_bulb_series = temp_data.apply(lambda row: psy.GetTWetBulbFromRelHum(row['tmpc'], row['relh'], row['alti']), axis=1)
    return twet_bulb_series