# IntensityLib
![WorkFlow](https://github.com/user-attachments/assets/efd9dfc4-8647-4c49-9917-158d2c645bc5)
This library provides dynamic average Carbon and Water Intensity time series data for the European Power Grid based on different calculation methods. It integrates a Flow Tracing Algorithm and can also derive the direct water consumption of data centers.
The picture shows the workflow of the framework. As input it takes coordinates or country codes as well as a start and end timestamp and returns the different time series data for the different objectives.

An exemplary use case can be found in usecase_europe.ipynb


## Requirements
The framework dependencies can be installed using:

pip install xarray reverse_geocoder itertools numpy pandas scipy psychrolib entsoe-py
