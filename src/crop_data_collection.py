import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_crop_data(start_date, end_date, location):
    date_range = pd.date_range(start=start_date, end=end_date)

    # Define crop types and their growth stages
    crop_types = ['Corn', 'Soybeans', 'Wheat', 'Cotton', 'Rice']
    growth_stages = ['Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Mature']

    # Define pest management practices
    pest_management = ['Chemical Control', 'Biological Control', 'Cultural Control', 'No Control']

    crop_data = pd.DataFrame({
        'Date': date_range,
        'Location': location,
        'Crop_Type': np.random.choice(crop_types, len(date_range)),
        'Growth_Stage': np.random.choice(growth_stages, len(date_range)),
        'Pest_Management': np.random.choice(pest_management, len(date_range))
    })

    # Add seasonal planting schedules (simplified)
    crop_data['Planting_Season'] = crop_data['Date'].dt.month.map({
        1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
        11: 'Fall', 12: 'Winter'
    })

    return crop_data

def save_crop_data(df, filename='historical_crop_data.csv'):
    df.to_csv(f'data/{filename}', index=False)
    print(f"Crop data has been saved to 'data/{filename}'")

def main():
    # Define parameters
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    locations = ['California', 'Iowa', 'Texas', 'Illinois', 'Nebraska']

    # Generate and combine crop data for all locations
    all_crop_data = pd.concat([
        generate_crop_data(start_date, end_date, location)
        for location in locations
    ])

    # Save the combined crop data
    save_crop_data(all_crop_data)

    # Display a sample of the data
    print("\nSample of the generated crop data:")
    print(all_crop_data.head())

if __name__ == "__main__":
    main()
