import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import os

def load_and_prepare_data():
    pest_data = pd.read_csv('data/cleaned_pest_data.csv')
    combined_data = pd.read_csv('data/combined_weather_soil_data.csv')

    merged_data = pd.merge(pest_data, combined_data, on=['Date', 'State'])

    print("Columns in merged_data:", merged_data.columns)

    # Use 'Month_x' as the 'Month' feature
    merged_data['Month'] = merged_data['Month_x']

    features = ['Temperature', 'Humidity', 'Rainfall', 'Wind_Speed', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Month', 'Soil_Moisture_Category']
    X = pd.get_dummies(merged_data[features], columns=['Soil_Moisture_Category'])
    y = merged_data['Pest_x']  # Use 'Pest_x' instead of 'Pest'

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    return X, y_encoded, le

def train_and_evaluate_model(X, y):
    # Split the data into training and testing sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    model = xgb.XGBClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Perform cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\nTest set evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    return model, None  # We don't need to return label_encoder anymore

def save_model_and_encoder(model, label_encoder):
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, 'xgboost_model.joblib'))
    joblib.dump(label_encoder, os.path.join(model_dir, 'label_encoder.joblib'))
    print(f"Model and label encoder saved in {model_dir} directory.")

if __name__ == "__main__":
    X, y, le = load_and_prepare_data()
    trained_model, _ = train_and_evaluate_model(X, y)
    save_model_and_encoder(trained_model, le)
    print("\nModel training, evaluation, and saving completed.")
