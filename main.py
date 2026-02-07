import os
import pandas as pd

RAW_DATA_DIR = "data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Load dataset (example: Kaggle student performance CSV)
df = pd.read_csv("student.csv")   # uploaded / downloaded dataset

# Save to DVC-tracked folder
df.to_csv(os.path.join(RAW_DATA_DIR, "student.csv"), index=False)

print("Data ingestion completed.")

import os
import pandas as pd

RAW_PATH = "data/raw/student.csv"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH)

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

df.to_csv(os.path.join(PROCESSED_DIR, "clean_data.csv"), index=False)

print("Data preprocessing completed.")

import os
import pandas as pd

INPUT_PATH = "data/processed/clean_data.csv"
OUTPUT_PATH = "data/processed/features.csv"

df = pd.read_csv(INPUT_PATH)

# Target column (example)
TARGET = "final_grade"

X = df.drop(columns=[TARGET])
y = df[TARGET]

final_df = pd.concat([X, y], axis=1)
final_df.to_csv(OUTPUT_PATH, index=False)

print("Feature engineering completed.")

import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/processed/features.csv")

X = df.drop(columns=["final_grade"])
y = df["final_grade"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params["data"]["test_size"], random_state=42
)

with mlflow.start_run():
    model = LogisticRegression(max_iter=params["model"]["max_iter"])
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "model")

print("Model training completed.")

