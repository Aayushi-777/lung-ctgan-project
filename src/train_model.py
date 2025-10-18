import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.ctgan_generator import generate_synthetic_data


def train_model(
    train_path="data/processed/train.csv",
    synthetic_path="data/processed/synthetic_minority.csv",
    augmented_path="data/processed/train_augmented.csv",
    cancer_model_path="results/random_forest_cancer.pkl",
    stage_model_path="results/random_forest_stage.pkl",
):
    # Step 1: Generate synthetic data
    synthetic_data = generate_synthetic_data(
        train_path=train_path,
        output_path=synthetic_path,
        epochs=50,
        sample_frac=0.3
    )

    # Step 2: Load datasets
    real_data = pd.read_csv(train_path)
    synthetic_data = pd.read_csv(synthetic_path)
    synthetic_data = synthetic_data.reindex(columns=real_data.columns, fill_value=0)

    # Merge and shuffle
    augmented_data = pd.concat([real_data, synthetic_data], ignore_index=True).sample(frac=1, random_state=42)
    os.makedirs(os.path.dirname(augmented_path), exist_ok=True)
    augmented_data.to_csv(augmented_path, index=False)
    print(f"[INFO] Augmented data saved: {augmented_path}, shape={augmented_data.shape}")

    # -----------------------------
    # MODEL 1: Cancer Prediction
    # -----------------------------
    X_cancer = augmented_data.drop(columns=["cancer", "stage"], errors="ignore")
    y_cancer = augmented_data["cancer"]

    cat_cols = X_cancer.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_cancer.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor_cancer = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    clf_cancer = Pipeline([
        ("preprocessor", preprocessor_cancer),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X_cancer, y_cancer, test_size=0.2, random_state=42, stratify=y_cancer)
    clf_cancer.fit(X_train, y_train)

    cancer_metrics = {
        "accuracy": accuracy_score(y_test, clf_cancer.predict(X_test)),
        "precision": precision_score(y_test, clf_cancer.predict(X_test), average="weighted"),
        "recall": recall_score(y_test, clf_cancer.predict(X_test), average="weighted"),
        "f1_score": f1_score(y_test, clf_cancer.predict(X_test), average="weighted"),
    }

    print(f"[INFO] Cancer Model Metrics: {cancer_metrics}")
    joblib.dump(clf_cancer, cancer_model_path)
    print(f"[INFO] Cancer model saved: {cancer_model_path}")

    # -----------------------------
    # MODEL 2: Stage Prediction (only for cancer=1)
    # -----------------------------
    cancer_patients = augmented_data[augmented_data["cancer"] == 1]
    X_stage = cancer_patients.drop(columns=["stage"], errors="ignore")
    y_stage = cancer_patients["stage"]

    cat_cols_stage = X_stage.select_dtypes(include=["object"]).columns.tolist()
    num_cols_stage = X_stage.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor_stage = ColumnTransformer([
        ("num", StandardScaler(), num_cols_stage),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols_stage),
    ])

    clf_stage = Pipeline([
        ("preprocessor", preprocessor_stage),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_stage, y_stage, test_size=0.2, random_state=42, stratify=y_stage)
    clf_stage.fit(X_train_s, y_train_s)

    stage_metrics = {
        "accuracy": accuracy_score(y_test_s, clf_stage.predict(X_test_s)),
        "precision": precision_score(y_test_s, clf_stage.predict(X_test_s), average="weighted"),
        "recall": recall_score(y_test_s, clf_stage.predict(X_test_s), average="weighted"),
        "f1_score": f1_score(y_test_s, clf_stage.predict(X_test_s), average="weighted"),
    }

    print(f"[INFO] Stage Model Metrics: {stage_metrics}")
    joblib.dump(clf_stage, stage_model_path)
    print(f"[INFO] Stage model saved: {stage_model_path}")


if __name__ == "__main__":
    train_model()
