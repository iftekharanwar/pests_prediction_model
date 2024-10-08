import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_soil_data(start_date, end_date, location):
    date_range = pd.date_range(start=start_date, end=end_date)

    soil_data = pd.DataFrame({
        'Date': date_range,
        'Location': location,
        'pH': np.random.uniform(5.5, 7.5, len(date_range)),
        'Nitrogen': np.random.uniform(0, 100, len(date_range)),  # in ppm
        'Phosphorus': np.random.uniform(0, 100, len(date_range)),  # in ppm
        'Potassium': np.random.uniform(0, 300, len(date_range)),  # in ppm
        'Moisture': np.random.uniform(10, 40, len(date_range))  # in percentage
    })

    return soil_data

def save_soil_data(df, filename='historical_soil_data.csv'):
    df.to_csv(f'data/{filename}', index=False)
    print(f"Soil data has been saved to 'data/{filename}'")

def main():
    # Define parameters
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    locations = ['California', 'Iowa', 'Texas', 'Illinois', 'Nebraska']

    # Generate and combine soil data for all locations
    all_soil_data = pd.concat([
        generate_soil_data(start_date, end_date, location)
        for location in locations
    ])

    # Save the combined soil data
    save_soil_data(all_soil_data)

    # Display a sample of the data
    print("\nSample of the generated soil data:")
    print(all_soil_data.head())

if __name__ == "__main__":
    main()
