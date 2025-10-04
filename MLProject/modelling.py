import argparse
import pandas as pd
import joblib
import os
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

# Inisialisasi MLflow lokal
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Spotify Churn CI - Skilled")

with mlflow.start_run(run_name="RF_CI_Skilled", nested=True):
    # Training model
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Hitung metrik
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Logging manual
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan model secara lokal
    os.makedirs("artefak", exist_ok=True)
    model_path = f"artefak/model_n{args.n_estimators}_d{args.max_depth}.pkl"
    joblib.dump(model, model_path)

    # Simpan model ke MLflow
    mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"Model tersimpan di: {model_path}")
    print(f"Akurasi: {acc:.4f}")