import pandas as pd
import joblib
import json
import yaml
from sklearn.linear_model import Ridge 

# 1. Load params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

model_name = params["model"]["name"]
alpha = params["model"]["alpha"]
features = params["data"]["features"]
target = params["data"]["target"]

# 2. Load data
data_path = "data/raw/houses.csv"
df = pd.read_csv(data_path)

# 3. Preprocessing
X = df[features]
y = df[target]

# 4. Train model
if model_name == "linear_regression":
    model = Ridge(alpha=alpha)  # Ridge bisa menerima alpha
else:
    raise ValueError(f"Unsupported model: {model_name}")

model.fit(X, y)

# 5. Save model
model_path = "models/house_predictor.joblib"
joblib.dump(model, model_path)

# 6. Evaluate
score = model.score(X, y)
metrics = {"r2_score": score}

# 7. Save metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"Model trained! R2 Score: {score:.4f}")
