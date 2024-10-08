import xgboost as xgb
import numpy as np

# Create a small dataset
X = np.random.rand(10, 5)
y = np.random.randint(2, size=10)

# Try to create an XGBoost classifier
model = xgb.XGBClassifier(n_estimators=10, use_label_encoder=False)

# Try to fit the model
model.fit(X, y)

print("XGBoost is working correctly with OpenMP!")
