import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(filepath, target_column):
    # Load dataset
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical columns (if any)
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Encode target if categorical
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, scaler

