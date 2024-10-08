import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the data
pest_data = pd.read_csv('data/cleaned_pest_data.csv')
weather_data = pd.read_csv('data/cleaned_weather_data.csv')
soil_data = pd.read_csv('data/cleaned_soil_data.csv')

# Merge the datasets
merged_data = pd.merge(pest_data, weather_data, on=['Date', 'Location'], suffixes=('', '_weather'))
merged_data = pd.merge(merged_data, soil_data, on=['Date', 'Location'], suffixes=('', '_soil'))

# Remove duplicate columns and columns with suffixes indicating they are duplicates
columns_to_remove = [col for col in merged_data.columns if '_weather' in col or '_soil' in col]
merged_data.drop(columns=columns_to_remove, inplace=True)

# Feature engineering
merged_data['Season'] = pd.to_datetime(merged_data['Date']).dt.month.map({1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring', 6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall', 11: 'Fall', 12: 'Winter'})
merged_data['Temperature_Humidity_Ratio'] = merged_data['Temperature'] / merged_data['Humidity']
merged_data['Rainfall_Wind_Interaction'] = merged_data['Rainfall'] * merged_data['Wind_Speed']
merged_data['Soil_Fertility_Index'] = (merged_data['Nitrogen'] + merged_data['Phosphorus'] + merged_data['Potassium']) / 3

# Encode categorical variables
le = LabelEncoder()
merged_data['Pest'] = le.fit_transform(merged_data['Pest'])
categorical_columns = ['Season', 'Crop_Affected', 'Life_Cycle_Stage', 'Infestation_Level']
if 'Soil_Moisture_Category' in merged_data.columns:
    categorical_columns.append('Soil_Moisture_Category')

# Convert categorical columns to 'category' dtype
for col in categorical_columns:
    merged_data[col] = merged_data[col].astype('category')

merged_data = pd.get_dummies(merged_data, columns=categorical_columns)

# Prepare features and target
features = ['Temperature', 'Humidity', 'Rainfall', 'Wind_Speed', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Temperature_Humidity_Ratio', 'Rainfall_Wind_Interaction', 'Soil_Fertility_Index'] + [col for col in merged_data.columns if col.startswith(('Season_', 'Crop_Affected_', 'Life_Cycle_Stage_', 'Infestation_Level_', 'Soil_Moisture_Category_'))]
X = merged_data[features]
y = merged_data['Pest']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = XGBClassifier(random_state=42, enable_categorical=True)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Save the improved model
joblib.dump(model, 'model/improved_xgboost_model.joblib')
joblib.dump(le, 'model/improved_label_encoder.joblib')
print('\nImproved model and label encoder saved.')
