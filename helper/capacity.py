import pandas as pd
import helper.entsoe_wrapper as entsoe_wrapper
import numpy as np
from entsoe import EntsoePandasClient


"""
To do is a yearly differentiation between the installed capacity
"""

def get_remaining_capacity_per_generation_type(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    #exception for Germany since only the combined zone DE_LU has data for missing capacity
    if country == 'DE':
        country = 'DE_LU'
    missing_capa=entsoe_wrapper.get_missing_capacity(country=country, start=start, end=end)
    installed_capa=entsoe_wrapper.get_installed_capacity(country=country, start=start, end=end)
    missing_capa = missing_capa.reindex(columns=installed_capa.columns, fill_value=0)
    start_year=start.year
    end_year=end.year
    remaining_capa=pd.DataFrame(index=missing_capa.index, columns=missing_capa.columns)
    
    for i,year in enumerate(range(start_year, end_year + 1)):
        yearly_installed_capa = installed_capa.loc[installed_capa.index == year]
        yearly_missing_capa = missing_capa.loc[missing_capa.index.year == year]
        yearly_missing_capa = yearly_missing_capa.reindex(columns=installed_capa.columns, fill_value=0)
        yearly_remaining_capa = -yearly_missing_capa.subtract(yearly_installed_capa.iloc[0], axis=1)
        remaining_capa.loc[yearly_missing_capa.index] = yearly_remaining_capa
    #drop all columns with all nan or 0
    remaining_capa=remaining_capa.dropna(axis=1, how='all')
    remaining_capa=remaining_capa.loc[:, (remaining_capa != 0).any(axis=0)]
    #drop all where column has a value below 0
    remaining_capa=remaining_capa.loc[:, (remaining_capa >= 0).all(axis=0)]
    return remaining_capa

def get_usage_percentage_per_generation_type(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    
    remaining_capa=get_remaining_capacity_per_generation_type(country=country, start=start, end=end)
    generation=entsoe_wrapper.get_generation_data_1h(country=country, start=start, end=end)
    # Drop columns containing 'Consumption' in their name
    generation.drop(columns=[col for col in generation.columns if 'Consumption' in col], inplace=True)
    
    # Clean up column names
    generation.columns = generation.columns.str.replace('_Actual Aggregated', '')
    # Calculate usage percentage
    usage_percentage = 1- (remaining_capa - generation) / remaining_capa
    usage_percentage.replace(np.nan, 0, inplace=True)
    return usage_percentage

def get_usage_percentage_variable_generation(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    generation=entsoe_wrapper.get_generation_data_1h(country=country, start=start, end=end)
    # Drop columns containing 'Consumption' in their name
    generation.drop(columns=[col for col in generation.columns if 'Consumption' in col], inplace=True)
    
    # Clean up column names
    generation.columns = generation.columns.str.replace('_Actual Aggregated', '')
    if country =='IE':
        generation=generation.ffill()
    
    remaining_capa=get_remaining_capacity_per_generation_type(country=country, start=start, end=end)
    generation=generation.reindex(columns=remaining_capa.columns, fill_value=0)

    #Calculate the usage percentage of all variable generation units
    units=['Biomass','Fossil Brown coal/Lignite','Fossil Coal-derived gas','Fossil Gas','Fossil Hard coal','Fossil Oil','Fossil Oil shale','Fossil Peat','Hydro Pumped Storage','Hydro Water Reservoir','Nuclear','Other','Waste']
    units_intersect=remaining_capa.columns.intersection(units)
    variable_gen_capa=remaining_capa[units_intersect].sum(axis=1)
    variable_gen=generation[units_intersect].sum(axis=1)
    variable_gen_usage_percentage=1-(variable_gen_capa-variable_gen)/variable_gen_capa
    return variable_gen_usage_percentage

    
def get_missing_capacity_percentage_per_generation_type(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    installed_capa=entsoe_wrapper.get_installed_capacity(country=country, start=start, end=end)
    missing_capa=entsoe_wrapper.get_missing_capacity(country=country, start=start, end=end)
    # Reindex installed and missing capacity to match generation columns
    missing_capa = missing_capa.reindex(columns=installed_capa.columns, fill_value=0)
    
    missing_capa_percentage=pd.DataFrame(index=missing_capa.index, columns=missing_capa.columns)
    
    for i,year in enumerate(range(start_year, end_year + 1)):
        yearly_installed_capa = installed_capa.loc[installed_capa.index.year == year].iloc[i]
        yearly_missing_capa = missing_capa.loc[missing_capa.index.year == year]
        missing_capa_percentage.loc[yearly_missing_capa.index] = yearly_missing_capa / yearly_installed_capa
    return missing_capa_percentage

def get_missing_capacity_percentage(country: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    start_year=start.year
    end_year=end.year
    installed_capa=entsoe_wrapper.get_installed_capacity(country=country, start=start, end=end)
    missing_capa=entsoe_wrapper.get_missing_capacity(country=country, start=start, end=end)
    missing_capa_percentage=pd.DataFrame(index=missing_capa.index, columns=["Missing_Capacity_Percentage"])
    
    for i,year in enumerate(range(start_year, end_year + 1)):
        yearly_installed_capa = installed_capa.loc[installed_capa.index.year == year].iloc[i].sum()
        yearly_missing_capa = missing_capa.loc[missing_capa.index.year == year].sum(axis=1)
        missing_capa_percentage.loc[yearly_missing_capa.index] = yearly_missing_capa / yearly_installed_capa
    return missing_capa_percentage