from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'templates'), static_folder=os.path.join(os.path.dirname(__file__), '..', 'static'))

# Load the trained model and label encoder
model = joblib.load('model/improved_xgboost_model.joblib')
label_encoder = joblib.load('model/improved_label_encoder.joblib')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the form
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    rainfall = float(request.form['rainfall'])
    wind_speed = float(request.form['wind_speed'])
    crop_type = request.form['crop_type']

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'Temperature': [temperature],
        'Humidity': [humidity],
        'Rainfall': [rainfall],
        'Wind_Speed': [wind_speed],
        'pH': [7.0],  # Default value, adjust as needed
        'Nitrogen': [0.5],  # Default value, adjust as needed
        'Phosphorus': [0.5],  # Default value, adjust as needed
        'Potassium': [0.5],  # Default value, adjust as needed
        'Moisture': [50],  # Default value, adjust as needed
        'Crop_Affected': [crop_type],
        'Life_Cycle_Stage': ['Adult'],  # Default value, adjust as needed
        'Infestation_Level': ['Medium']  # Default value, adjust as needed
    })

    # Calculate derived features
    input_data['Temperature_Humidity_Ratio'] = input_data['Temperature'] / input_data['Humidity']
    input_data['Rainfall_Wind_Interaction'] = input_data['Rainfall'] * input_data['Wind_Speed']
    input_data['Soil_Fertility_Index'] = (input_data['Nitrogen'] + input_data['Phosphorus'] + input_data['Potassium']) / 3

    # Add season based on current month (you may need to import datetime)
    from datetime import datetime
    current_month = datetime.now().month
    season = 'Spring' if 3 <= current_month <= 5 else 'Summer' if 6 <= current_month <= 8 else 'Fall' if 9 <= current_month <= 11 else 'Winter'
    input_data['Season'] = season

    # Perform one-hot encoding for categorical variables
    input_data = pd.get_dummies(input_data, columns=['Crop_Affected', 'Life_Cycle_Stage', 'Infestation_Level', 'Season'])

    # Make sure all columns from training data are present
    expected_columns = ['Temperature', 'Humidity', 'Rainfall', 'Wind_Speed', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Moisture', 'Temperature_Humidity_Ratio', 'Rainfall_Wind_Interaction', 'Soil_Fertility_Index', 'Season_Fall', 'Season_Spring', 'Season_Summer', 'Season_Winter', 'Crop_Affected_Corn', 'Crop_Affected_Cotton', 'Crop_Affected_Rice', 'Crop_Affected_Soybeans', 'Crop_Affected_Wheat', 'Life_Cycle_Stage_Adult', 'Life_Cycle_Stage_Egg', 'Life_Cycle_Stage_Larva', 'Life_Cycle_Stage_Pupa', 'Infestation_Level_High', 'Infestation_Level_Low', 'Infestation_Level_Medium']
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match the training data
    input_data = input_data[expected_columns]

    # Make prediction
    prediction = model.predict(input_data)
    predicted_pest = label_encoder.inverse_transform(prediction)[0]

    # Get management strategies for the predicted pest
    strategies = get_management_strategies(predicted_pest)

    return render_template('result.html', pest=predicted_pest, strategies=strategies, pest_image=f"{predicted_pest.lower().replace(' ', '_')}.jpg")

def get_management_strategies(pest):
    # This is a simplified version. In a real-world scenario, you would have a more comprehensive database of strategies.
    strategies = {
        'Aphids': [
            'Use insecticidal soaps or neem oil',
            'Introduce natural predators like ladybugs',
            'Remove heavily infested plant parts'
        ],
        'Corn Rootworm': [
            'Implement crop rotation',
            'Use Bt corn varieties',
            'Apply soil insecticides during planting'
        ],
        'Japanese Beetle': [
            'Handpick beetles in small gardens',
            'Use pheromone traps to monitor populations',
            'Apply neem oil or pyrethrin-based insecticides'
        ],
        'Cutworms': [
            'Use protective collars around young plants',
            'Keep the area around plants free of weeds',
            'Apply diatomaceous earth around plants'
        ],
        'Armyworms': [
            'Monitor fields regularly for early detection',
            'Use pheromone traps to track moth populations',
            'Apply Bacillus thuringiensis (Bt) based products'
        ]
    }
    return strategies.get(pest, ['No specific strategies available for this pest.'])

if __name__ == '__main__':
    app.run(debug=True)
