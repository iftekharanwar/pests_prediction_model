import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

def load_and_prepare_data():
    # Load the preprocessed data
    pest_data = pd.read_csv('data/cleaned_pest_data.csv')
    combined_data = pd.read_csv('data/combined_weather_soil_data.csv')

    print("Pest data shape:", pest_data.shape)
    print("Combined data shape:", combined_data.shape)

    print("\nPest data date range:")
    print(pest_data['Date'].min(), "-", pest_data['Date'].max())
    print("\nCombined data date range:")
    print(combined_data['Date'].min(), "-", combined_data['Date'].max())

    # Convert 'Date' to datetime for both datasets
    pest_data['Date'] = pd.to_datetime(pest_data['Date'])
    combined_data['Date'] = pd.to_datetime(combined_data['Date'])

    # Merge pest data with combined weather and soil data based on 'Date' only
    merged_data = pd.merge(pest_data, combined_data, on=['Date'])

    print("\nMerged data shape:", merged_data.shape)

    if merged_data.empty:
        print("Warning: Merged dataset is empty. Check date ranges and formats.")
        return None, None, None, None, None

    # Prepare features and target
    features = ['Temperature', 'Humidity', 'Rainfall', 'Wind_Speed', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Month', 'Soil_Moisture_Category']
    X = pd.get_dummies(merged_data[features], columns=['Soil_Moisture_Category'])
    y = merged_data['Pest']

    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return train_test_split(X, y_encoded, test_size=0.2, random_state=42), le

def evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy:', accuracy)
    print(f'{model_name} Classification Report:')
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    return accuracy

def main():
    (X_train, X_test, y_train, y_test), label_encoder = load_and_prepare_data()

    if X_train is None:
        print("Data preparation failed. Exiting.")
        return

    models = [
        (DecisionTreeClassifier(random_state=42), "Decision Tree"),
        (RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest"),
        (xgb.XGBClassifier(random_state=42), "XGBoost")
    ]

    results = {}

    for model, name in models:
        accuracy = evaluate_model(model, X_train, X_test, y_train, y_test, label_encoder, name)
        results[name] = accuracy

    print('\nModel Comparison:')
    for name, accuracy in results.items():
        print(f'{name} Accuracy: {accuracy}')

    best_model = max(results, key=results.get)
    print(f'\nBest performing model: {best_model}')

if __name__ == "__main__":
    main()
