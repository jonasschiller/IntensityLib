import pandas as pd
import numpy as np


def extend_workload(df):
    df['Weekend']=df.index.weekday.isin([5,6])
    # Separate the data into weekdays and weekends
    weekday_data = df[df['Weekend'] == False]
    weekend_data = df[df['Weekend'] == True]

    # Calculate the number of weekdays and weekends in a year
    num_weekdays = 261 * 24  # 5 weekdays per week * 52 weeks + 1 extra weekday, each with 24 hours
    num_weekends = 104 * 24  # 2 weekends per week * 52 weeks, each with 24 hours

    # Repeat the data to match the number of weekdays and weekends in a year
    extended_weekday_data = pd.concat([weekday_data] * (num_weekdays // len(weekday_data)) + [weekday_data[:num_weekdays % len(weekday_data)]])
    extended_weekend_data = pd.concat([weekend_data] * (num_weekends // len(weekend_data)) + [weekend_data[:num_weekends % len(weekend_data)]])

    # Combine the extended weekday and weekend data
    extended_data = pd.concat([extended_weekday_data, extended_weekend_data])
    extended_data = extended_data * np.random.normal(1, 0.1, size=extended_data.shape)

    # Sort the data by index to maintain the order
    extended_data = extended_data.sort_index()

    # Reset the index to have a continuous date range
    extended_data.index = pd.date_range(start='2022-01-01', periods=365*24, freq='h')

    return extended_data