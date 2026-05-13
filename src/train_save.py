import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(BASE_DIR, "data", "train.csv")

model_path = os.path.join(BASE_DIR, "models", "model.pkl")
features_path = os.path.join(BASE_DIR, "models", "features.pkl")

# Load data
df = pd.read_csv(data_path)

# Select ONLY strong features
selected_features = [
    "Lot Area",
    "Year Built",
    "Overall Qual",
    "Gr Liv Area",
    "Total Bsmt SF",
    "Garage Cars"
]

df = df[selected_features + ["SalePrice"]]

X = df[selected_features]
y = df["SalePrice"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple preprocessing (NO ONE-HOT needed now)
preprocessor = SimpleImputer(strategy="median")

model = RandomForestRegressor(
    n_estimators=400,
    random_state=42
)

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
clf.fit(X_train, y_train)

# Save model + features
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

joblib.dump(clf, model_path)
joblib.dump(selected_features, features_path)

print("SUCCESS: Clean model trained + saved")