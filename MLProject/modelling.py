# modelling.py - untuk MLflow Project
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=10)
parser.add_argument("--dataset_path", type=str, default="spotify_churn_dataset_preprocessing")
args = parser.parse_args()

# Load dataset
X_train = pd.read_csv(f"{args.dataset_path}/X_train.csv")
X_test = pd.read_csv(f"{args.dataset_path}/X_test.csv")
y_train = pd.read_csv(f"{args.dataset_path}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{args.dataset_path}/y_test.csv").squeeze()

# Setup MLflow lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Spotify Churn Workflow CI")

mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RF_CI_Basic"):
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Model selesai dilatih. Akurasi: {acc:.4f}")