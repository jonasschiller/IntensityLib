#This file contains the code required for the flow tracing algorithm on the european power grid
#It is based on the methodology used by electricityMap

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

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