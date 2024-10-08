import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import joblib
import os

from data_preprocessing import clean_dataset, create_features
from data_collection import generate_pest_data
from model_training import train_and_evaluate_model

def fetch_new_data(num_records=1000):
    # Set the date range for all datasets
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    locations = ['California', 'Iowa', 'Texas', 'Illinois', 'Nebraska']

    # Simulate fetching new pest data
    new_pest_data = generate_pest_data(num_records, start_date, end_date)
    new_pest_data['Location'] = new_pest_data['State']

    # Simulate fetching new weather data
    new_weather_data = pd.DataFrame({
        'Date': date_range.repeat(len(locations)),
        'Location': np.tile(locations, len(date_range)),
        'Temperature': np.random.uniform(10, 35, len(date_range) * len(locations)),
        'Humidity': np.random.uniform(30, 90, len(date_range) * len(locations)),
        'Rainfall': np.random.uniform(0, 50, len(date_range) * len(locations)),
        'Wind_Speed': np.random.uniform(0, 30, len(date_range) * len(locations))
    })

    # Simulate fetching new soil data
    new_soil_data = pd.DataFrame({
        'Date': date_range.repeat(len(locations)),
        'Location': np.tile(locations, len(date_range)),
        'pH': np.random.uniform(5.5, 7.5, len(date_range) * len(locations)),
        'Nitrogen': np.random.uniform(0, 100, len(date_range) * len(locations)),
        'Phosphorus': np.random.uniform(0, 100, len(date_range) * len(locations)),
        'Potassium': np.random.uniform(0, 100, len(date_range) * len(locations)),
        'Moisture': np.random.uniform(10, 50, len(date_range) * len(locations))
    })

    return new_pest_data, new_weather_data, new_soil_data

def update_model():
    print("Fetching new data...")
    new_pest_data, new_weather_data, new_soil_data = fetch_new_data()

    print("Preprocessing new data...")
    cleaned_pest_data = clean_dataset(new_pest_data)
    cleaned_weather_data = clean_dataset(new_weather_data)
    cleaned_soil_data = clean_dataset(new_soil_data)

    pest_data, weather_data, soil_data, _, combined_data = create_features(cleaned_pest_data, cleaned_weather_data, cleaned_soil_data, pd.DataFrame())

    print("Retraining model with new data...")
    features = ['Temperature', 'Humidity', 'Rainfall', 'Wind_Speed', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Month', 'Soil_Moisture_Category']
    X = pd.get_dummies(combined_data[features], columns=['Soil_Moisture_Category'])
    y = combined_data['Pest']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print("X shape:", X.shape)
    print("y shape:", y_encoded.shape)
    print("X columns:", X.columns)
    print("X head:", X.head())

    new_model, new_label_encoder = train_and_evaluate_model(X, y_encoded)

    print("Saving updated model...")
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(new_model, os.path.join(model_dir, 'xgboost_model.joblib'))
    joblib.dump(new_label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))

    print("Model update completed successfully.")

if __name__ == "__main__":
    update_model()
