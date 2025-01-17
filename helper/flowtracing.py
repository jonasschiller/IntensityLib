#This file contains the code required for the flow tracing algorithm on the european power grid
#It is based on the methodology used by electricityMap

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import helper.entsoe_wrapper as entsoe_wrapper
import os

def solve_linear_system(A, b):
    """
    Solve the linear system Ax = b for a single timestep.
    """
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.full(b.shape, np.nan)  # Return NaNs if the system is singular

def solve_linear_systems_parallel(A, b, valid_countries_list, n_countries):
    """
    Solve the linear systems Ax = b in parallel for all timesteps, adjusting for NaN values.

    Parameters:
    - A: list of numpy arrays of shape (n, n)
    - b: list of numpy arrays of shape (n,)
    - valid_countries_list: list of lists containing valid country indices for each timestep
    - n_countries: total number of countries

    Returns:
    - x: numpy array of shape (timesteps, n) containing the solutions
    """
    timesteps = len(A)
    
    # Initialize the result array
    x = np.full((timesteps, n_countries), np.nan)
    
    # Use ThreadPoolExecutor to parallelize the computation
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(solve_linear_system, A, b))
    
    # Collect the results
    for t, result in enumerate(results):
        x[t, :] = result
    
    return x


def get_A(gen_dict, netimp_dict, countryCodes, timesteps):
    """
    Construct the matrix A for the linear system Ax = b for all timesteps.

    Parameters:
    - gen_dict: dictionary containing generation data for each country
    - netimp_dict: dictionary containing net import data for each country
    - countryCodes: list of country codes
    - timesteps: number of timesteps

    Returns:
    - A: numpy array of shape (timesteps, n, n) containing the matrices for each timestep
    """
    n = len(countryCodes)
    A = np.zeros((timesteps, n, n))

    for i, country_code in enumerate(countryCodes):
        # Calculate the sum of imports for the current country, considering only the relevant columns
        import_sums = netimp_dict[country_code][list(set(countryCodes).intersection(netimp_dict[country_code].columns))].sum(axis=1).iloc[:timesteps].values
        
        # Set the diagonal element for the current country
        A[:, i, i] = gen_dict[country_code].sum(axis=1).iloc[:timesteps].values + import_sums
        
        # Adjust the diagonal element by adding exports
        for key in netimp_dict[country_code].columns:
            if key in countryCodes:
                help = netimp_dict[country_code][key].iloc[:timesteps].values.copy()
                help[help > 0] = 0
                A[:, i, i] -= help

        # Set the off-diagonal elements for the current country with the negative values of the imports
        for s, country_code2 in enumerate(countryCodes):
            if i != s:
                help = netimp_dict[country_code].get(country_code2, pd.Series([0])).iloc[:timesteps].values.copy()
                help[help < 0] = 0
                A[:, i, s] = -help

    return A
    
def get_b(gen_dict, gen_type, countryCodes, timesteps):
    """
    Construct the vector b for the linear system Ax = b for all timesteps.

    Parameters:
    - gen_dict: dictionary containing generation data for each country
    - gen_type: type of generation to consider (e.g., 'solar', 'wind')
    - countryCodes: list of country codes
    - timesteps: number of timesteps

    Returns:
    - b: numpy array of shape (timesteps, n) containing the vectors for each timestep
    """
    # Initialize the result array
    b = np.array([
        # Extract the generation data for the specified type and country, limited to the given timesteps
        gen_dict[country_code][gen_type].iloc[:timesteps].values
        for country_code in countryCodes
    ]).T  # Transpose to get the correct shape (timesteps, n)
    
    return b

def get_data_flow(countryCodes,start_utc,end_utc):
        #   Load data
    gen_dict,gen_types=entsoe_wrapper.get_generation_dict(countryCodes,start_utc,end_utc,imputation_type="no")
    gen_types.difference_update(["Hydro Pumped Storage_Actual Consumption"])
    import_dict=entsoe_wrapper.get_import_dict(countryCodes,start_utc,end_utc,imputation_type="no") 
    export_dict=entsoe_wrapper.get_export_dict(countryCodes,start_utc,end_utc,imputation_type="no")
    netimp_dict={}
    gen_dict_without_con=gen_dict.copy()
    for country_code in countryCodes:
        #fill nan values with the last known value
        gen_dict[country_code].ffill(inplace=True)
        #Drop columns that monitor consumption (irrelevant for the model)
        gen_dict_without_con[country_code] = gen_dict[country_code].loc[:, ~gen_dict[country_code].columns.str.contains("Actual Consumption")]
        #fill nan values with the last known value
        export_dict[country_code]=export_dict[country_code].drop(columns=["sum"]).ffill()
        #fill nan values with the last known value
        import_dict[country_code]=import_dict[country_code].drop(columns=["sum"]).ffill()
        #calculate net imports
        netimp_dict[country_code]=import_dict[country_code]-export_dict[country_code]
    return gen_dict_without_con,netimp_dict,gen_types

def check_sum(q,gen_types):
    sum=0
    for gen_type in gen_types:
        q[gen_type]=np.nan_to_num(q[gen_type],0)
        sum+=q[gen_type]
    #print number of values that are not equal to 1
    print(np.count_nonzero(sum!=1))


def calculate_flowtrace_all(country,year,countryCodes=["AT","PT","ES","FR","IT","GR","ME","BG","RO","RS","HU","SK","SI","CZ","BE","NL","EE","LV","LT","FI","NO","SE","DK","PL","DE","IE"]):
    gen_dict,netimp_dict,gen_types=get_data_flow(countryCodes,pd.Timestamp(str(year)+"-01-01",tz="UTC"),pd.Timestamp(str(year+1)+"-01-01",tz="UTC"))
    timesteps = len(gen_dict[country])
    A = get_A(gen_dict, netimp_dict, countryCodes, timesteps)
    q = {}
    
    for gen_type in gen_types:
        b = get_b(gen_dict,gen_type, countryCodes, timesteps)
        q[gen_type] = solve_linear_systems_parallel(A, b, countryCodes, len(countryCodes))
    
    gen_dict_flow=get_flow_dict(q,gen_dict,gen_types,countryCodes)

    return gen_dict_flow

def calculate_flowtrace_country(country,year,countryCodes=["AT","PT","ES","FR","IT","GR","ME","BG","RO","RS","HU","SK","SI","CZ","BE","NL","EE","LV","LT","FI","NO","SE","DK","PL","DE","IE"]):
    gen_dict_flow=calculate_flowtrace_all(country,year,countryCodes)
    for country in countryCodes:
        filename = os.path.join(entsoe_wrapper.CACHE_DIR, "GenerationFlowTrace\\Gen_" + country + "_" + str(year) + ".csv")
        gen_dict_flow[country].to_csv(filename, index=True)
    return gen_dict_flow[country]




def get_flow_dict(q,gen_dict,gen_types,countryCodes):
    # Translate q back into generation amounts per country
    gen_dict_flow=gen_dict.copy()
    for country in countryCodes:
        gen_dict_flow[country]=pd.DataFrame(index=gen_dict[country].index, columns=list(gen_types))
        for gen_type in gen_types:
            gen_dict_flow[country][gen_type]=q[gen_type][:,countryCodes.index(country)]*gen_dict[country].sum(axis=1)
    # Replace the null entries where missing values were imputed with the original values
    zero_indices={}
    for country in countryCodes:
        zero_indices[country]=np.where(gen_dict_flow[country].sum(axis=1)==0)[0]
        gen_dict_flow[country].iloc[zero_indices[country]]=gen_dict[country].iloc[zero_indices[country]]
    return gen_dict_flow

def get_generation_flowtrace(country,start,end):
    start_year=start.year
    end_year=end.year
    years=np.arange(start_year,end_year+1)
    data=None
    
    for year in years:
        filename = os.path.join(entsoe_wrapper.CACHE_DIR, "GenerationFlowTrace\\Gen_" + country + "_" + str(year) + ".csv")
        # Check if the file exists at all
        if os.path.exists(filename):
            data_helper= pd.read_csv(filename,index_col=0,parse_dates=True)
        else:
            data_helper = calculate_flowtrace_country(country, year)
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