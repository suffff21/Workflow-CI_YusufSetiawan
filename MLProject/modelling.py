import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

with mlflow.start_run():
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluasi model
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Logging manual ke MLflow
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Simpan model ke file lokal
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)

    # Log artefak ke MLflow
    mlflow.log_artifact(model_filename)

print("Training selesai dan model berhasil disimpan serta di-log ke MLflow.")