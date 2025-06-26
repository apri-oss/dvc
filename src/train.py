import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import json
import sys

# 1. Load data
data_path = "data/raw/houses.csv"
df = pd.read_csv(data_path)

# 2. Preprocessing
X = df[['size', 'bedrooms']]
y = df['price']

# 3. Train model
model = LinearRegression()
model.fit(X, y)

# 4. Save model
model_path = "models/house_predictor.joblib"
joblib.dump(model, model_path)

# 5. Evaluate
score = model.score(X, y)
metrics = {"r2_score": score}

# 6. Save metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Model trained! R2 Score: {score:.4f}")