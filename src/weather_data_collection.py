import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def fetch_weather_data(start_date, end_date, location):
    # In a real scenario, we would use an API like OpenWeatherMap or NOAA
    # For this simulation, we'll generate random weather data
    date_range = pd.date_range(start=start_date, end=end_date)

    weather_data = pd.DataFrame({
        'Date': date_range,
        'Temperature': np.random.uniform(10, 35, len(date_range)),  # in Celsius
        'Humidity': np.random.uniform(30, 90, len(date_range)),  # in percentage
        'Rainfall': np.random.exponential(5, len(date_range)),  # in mm
        'Wind_Speed': np.random.uniform(0, 20, len(date_range)),  # in km/h
    })

    # Add location information
    weather_data['Location'] = location

    # Calculate drought occurrences (simplified: 7 consecutive days with rainfall < 1mm)
    weather_data['Drought'] = (
        (weather_data['Rainfall'] < 1)
        .rolling(window=7, center=False)
        .sum()
        .ge(7)
        .astype(int)
    )

    return weather_data

def save_weather_data(df, filename='historical_weather_data.csv'):
    df.to_csv(f'data/{filename}', index=False)
    print(f"Weather data has been saved to 'data/{filename}'")

def main():
    # Define parameters
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    locations = ['California', 'Iowa', 'Texas', 'Illinois', 'Nebraska']

    # Fetch and combine weather data for all locations
    all_weather_data = pd.concat([
        fetch_weather_data(start_date, end_date, location)
        for location in locations
    ])

    # Save the combined weather data
    save_weather_data(all_weather_data)

    # Display a sample of the data
    print("\nSample of the collected weather data:")
    print(all_weather_data.head())

if __name__ == "__main__":
    main()
