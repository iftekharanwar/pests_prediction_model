import pandas as pd
import numpy as np

def clean_dataset(df):
    # Handle missing values
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64']:
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])

    # Handle outliers using IQR method for numerical columns
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)

    return df

def create_features(pest_data, weather_data, soil_data, crop_data):
    print("Pest data shape:", pest_data.shape)
    print("Weather data shape:", weather_data.shape)
    print("Soil data shape:", soil_data.shape)

    print("Pest data date range:", pest_data['Date'].min(), "-", pest_data['Date'].max())
    print("Weather data date range:", weather_data['Date'].min(), "-", weather_data['Date'].max())
    print("Soil data date range:", soil_data['Date'].min(), "-", soil_data['Date'].max())

    print("Pest data states:", pest_data['State'].unique())
    print("Weather data locations:", weather_data['Location'].unique())
    print("Soil data locations:", soil_data['Location'].unique())

    # Add seasonal indicators
    for df in [pest_data, weather_data, soil_data]:
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['Season'] = pd.cut(df['Date'].dt.month, bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])

    # Combine soil and weather conditions
    print("Merging weather and soil data...")
    combined_data = pd.merge(weather_data, soil_data, on=['Date', 'Location'], suffixes=('_weather', '_soil'))
    print("Combined data shape after weather and soil merge:", combined_data.shape)

    # Check if 'Moisture_soil' exists, if not, use 'Moisture'
    moisture_column = 'Moisture_soil' if 'Moisture_soil' in combined_data.columns else 'Moisture'
    combined_data['Soil_Moisture_Category'] = pd.cut(combined_data[moisture_column], bins=[0, 20, 40, 60, 100], labels=['Very Dry', 'Dry', 'Moist', 'Wet'])

    combined_data['Weather_Soil_Combination'] = combined_data['Soil_Moisture_Category'].astype(str) + '_' + (combined_data['Temperature'] > 25).map({True: 'Hot', False: 'Cool'})

    # Ensure 'Month' column is in combined_data
    if 'Month' not in combined_data.columns:
        combined_data['Month'] = combined_data['Date'].dt.month

    # Merge pest_data with combined_data
    print("Merging pest data with combined data...")
    combined_data = pd.merge(combined_data, pest_data[['Date', 'State', 'Pest', 'Month']], left_on=['Date', 'Location'], right_on=['Date', 'State'], how='inner', suffixes=('', '_pest'))
    print("Final combined data shape:", combined_data.shape)
    print("Columns in combined data:", combined_data.columns)

    if combined_data.shape[0] == 0:
        print("Warning: No matching data found between pest data and weather/soil data.")
        print("Sample of pest data:")
        print(pest_data.head())
        print("Sample of combined weather/soil data:")
        print(combined_data.head())

    return pest_data, weather_data, soil_data, crop_data, combined_data

def main():
    # Load all datasets
    pest_data = pd.read_csv('data/historical_pest_data.csv')
    weather_data = pd.read_csv('data/historical_weather_data.csv')
    soil_data = pd.read_csv('data/historical_soil_data.csv')
    crop_data = pd.read_csv('data/historical_crop_data.csv')

    # Clean datasets
    pest_data = clean_dataset(pest_data)
    weather_data = clean_dataset(weather_data)
    soil_data = clean_dataset(soil_data)
    crop_data = clean_dataset(crop_data)

    # Create features
    pest_data, weather_data, soil_data, crop_data, combined_data = create_features(pest_data, weather_data, soil_data, crop_data)

    # Save cleaned and feature-enhanced datasets
    pest_data.to_csv('data/cleaned_pest_data.csv', index=False)
    weather_data.to_csv('data/cleaned_weather_data.csv', index=False)
    soil_data.to_csv('data/cleaned_soil_data.csv', index=False)
    crop_data.to_csv('data/cleaned_crop_data.csv', index=False)
    combined_data.to_csv('data/combined_weather_soil_data.csv', index=False)

    print('Data cleaning, preprocessing, and feature creation completed. Enhanced datasets saved.')

    # Display sample of cleaned and feature-enhanced data
    print('\nSample of cleaned and feature-enhanced pest data:')
    print(pest_data.head())
    print('\nSample of cleaned and feature-enhanced weather data:')
    print(weather_data.head())
    print('\nSample of cleaned and feature-enhanced soil data:')
    print(soil_data.head())
    print('\nSample of cleaned and feature-enhanced crop data:')
    print(crop_data.head())
    print('\nSample of combined weather and soil data:')
    print(combined_data.head())

    # Display summary statistics
    print('\nSummary statistics of cleaned and feature-enhanced pest data:')
    print(pest_data.describe(include='all'))
    print('\nSummary statistics of cleaned and feature-enhanced weather data:')
    print(weather_data.describe(include='all'))
    print('\nSummary statistics of cleaned and feature-enhanced soil data:')
    print(soil_data.describe(include='all'))
    print('\nSummary statistics of cleaned and feature-enhanced crop data:')
    print(crop_data.describe(include='all'))
    print('\nSummary statistics of combined weather and soil data:')
    print(combined_data.describe(include='all'))

if __name__ == "__main__":
    main()
