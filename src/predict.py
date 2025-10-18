import joblib
import pandas as pd

CANCER_MODEL_PATH = "results/random_forest_cancer.pkl"
STAGE_MODEL_PATH = "results/random_forest_stage.pkl"
TRAIN_PATH = "data/processed/train.csv"

def predict(input_data):
    # Load models
    clf_cancer = joblib.load(CANCER_MODEL_PATH)
    clf_stage = joblib.load(STAGE_MODEL_PATH)
    train_df = pd.read_csv(TRAIN_PATH)

    # Prepare feature columns (exclude target columns)
    feature_columns = train_df.drop(columns=["cancer", "stage"]).columns.tolist()
    df = pd.DataFrame(input_data).reindex(columns=feature_columns, fill_value=0)

    # Predict cancer
    cancer_pred = clf_cancer.predict(df)
    cancer_proba = clf_cancer.predict_proba(df)[:, 1]  # Probability of having cancer

    df["cancer_prediction"] = cancer_pred
    df["cancer_probability"] = cancer_proba.round(2)

    # Predict stage only for cancer-positive patients
    df["predicted_stage"] = 0
    has_cancer = df["cancer_prediction"] == 1
    if has_cancer.any():
        # Copy data for stage prediction and include 'cancer' column
        stage_df = df.loc[has_cancer].copy()
        stage_df["cancer"] = 1  # required for stage model
        stage_pred = clf_stage.predict(stage_df)
        df.loc[has_cancer, "predicted_stage"] = stage_pred

    return df


if __name__ == "__main__":
    sample_input = [
        {"age": 65, "smoking": 1, "cough": 1, "chest_pain": 0},
        {"age": 45, "smoking": 1, "cough": 0, "chest_pain": 1},
    ]

    results = predict(sample_input)
    print(results)