# Pest Prediction Model

This project implements a pest prediction model for agriculture using machine learning techniques.

## Prerequisites

- Python 3.12.1
- pip (Python package installer)

## Installation

1. Clone the repository.

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

   Note: The `requirements.txt` file includes the latest versions of all dependencies compatible with Python 3.12.1 and macOS ARM64 architecture.

## Usage

1. Run data collection scripts:
   ```
   python src/data_collection.py
   python src/weather_data_collection.py
   python src/soil_data_collection.py
   python src/crop_data_collection.py
   ```

2. Preprocess the data:
   ```
   python src/data_preprocessing.py
   ```

3. Train the model:
   ```
   python src/model_training.py
   ```

4. Start the Flask application:
   ```
   FLASK_APP=src/app.py flask run
   ```

5. Open a web browser and navigate to `http://127.0.0.1:5000` to use the pest prediction interface.

   Note: The web interface features a centered design with a blurry greyish and blackish color scheme for improved user experience.

## Troubleshooting

If you encounter any issues, please check the following:

- Ensure you're using Python 3.12.1
- Make sure all required packages are installed correctly
- Check if the data files are present in the `data` directory
- Verify that the model files are present in the `model` directory

### macOS ARM64 Specific Instructions

If you're using a Mac with Apple Silicon (ARM64 architecture), follow these additional steps:

1. Install Homebrew if you haven't already:
   ```
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Install OpenMP:
   ```
   brew install libomp
   ```

3. Install XGBoost from source:
   ```
   pip install --no-binary :all: xgboost
   ```

4. Verify OpenMP installation:
   ```
   brew info libomp
   ```
   This should display information about the installed OpenMP package.

5. Verify XGBoost installation:
   ```
   python -c "import xgboost; print(xgboost.__version__)"
   ```
   This should print the installed XGBoost version without any errors.

If you encounter issues with XGBoost, try the following:

1. Set the following environment variables before running the application:
   ```
   export OPENBLAS_NUM_THREADS=1
   export OMP_NUM_THREADS=1
   ```

2. If you still face issues, try using a CPU-only version of XGBoost:
   ```
   pip uninstall xgboost
   pip install xgboost --install-option="--use-system-libxgboost" --no-binary :all:
   ```

For any other issues, please open an issue on the GitHub repository.

## Testing the Application

To ensure that the application is working correctly after making changes or setting up on a new system, follow these steps:

1. Run the data collection and preprocessing scripts:
   ```
   python src/data_collection.py
   python src/weather_data_collection.py
   python src/soil_data_collection.py
   python src/crop_data_collection.py
   python src/data_preprocessing.py
   ```

2. Train the model:
   ```
   python src/model_training.py
   ```

3. Start the Flask application in debug mode:
   ```
   FLASK_DEBUG=1 FLASK_APP=src/app.py flask run
   ```

4. Open a web browser and navigate to `http://127.0.0.1:5000`

5. Test the prediction functionality by entering sample data and verifying that you receive a prediction and management strategies.

6. Verify that the pest images (aphids, Japanese beetle, spider mite, grasshopper, corn rootworm, armyworms, and cutworm) are displayed correctly in the prediction results.

If you encounter any errors during this process, please check the console output for error messages and refer to the troubleshooting section above.
