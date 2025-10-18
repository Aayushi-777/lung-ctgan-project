import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def evaluate_model(
    augmented_path="data/processed/train_augmented.csv",
    model_path="results/random_forest.pkl",
    target_column="cancer"
):
    # Step 1: Load augmented dataset
    df = pd.read_csv(augmented_path)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Train-test split (same split logic as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 2: Load trained pipeline
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] Model not found at {model_path}")
    clf = joblib.load(model_path)

    # Step 3: Evaluate
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_score": float(f1_score(y_test, y_pred, average="weighted")),
    }

    print("[INFO] Evaluation metrics:", metrics)
    return metrics


if __name__ == "__main__":
    evaluate_model()






