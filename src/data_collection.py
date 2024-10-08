import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_pest_data(num_records=1000, start_date=None, end_date=None):
    # List of common agricultural pests
    pests = ['Aphids', 'Corn Rootworm', 'Japanese Beetle', 'Cutworms', 'Armyworms']

    # List of common crops
    crops = ['Corn', 'Soybeans', 'Wheat', 'Cotton', 'Rice']

    # List of states
    states = ['California', 'Iowa', 'Texas', 'Illinois', 'Nebraska']

    # Use provided date range or default to the year 2023
    if start_date is None or end_date is None:
        end_date = datetime(2023, 12, 31)
        start_date = datetime(2023, 1, 1)

    # Generate random data
    data = {
        'Date': [start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days)) for _ in range(num_records)],
        'Pest': np.random.choice(pests, num_records),
        'Life_Cycle_Stage': np.random.choice(['Egg', 'Larva', 'Pupa', 'Adult'], num_records),
        'Crop_Affected': np.random.choice(crops, num_records),
        'State': np.random.choice(states, num_records),
        'Infestation_Level': np.random.choice(['Low', 'Medium', 'High'], num_records)
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Sort by date
    df = df.sort_values('Date')

    return df

def save_pest_data(df, filename='historical_pest_data.csv'):
    # Save the data to a CSV file
    df.to_csv(f'data/{filename}', index=False)
    print(f"Historical pest data has been saved to 'data/{filename}'")

def fetch_pest_data():
    # In a real scenario, this function would fetch data from an API or database
    # For now, we'll generate simulated data
    pest_data = generate_pest_data()

    # Save the data
    save_pest_data(pest_data)

    # Display a sample of the data
    print("\nSample of the collected data:")
    print(pest_data.head())

    return pest_data

if __name__ == "__main__":
    fetch_pest_data()
