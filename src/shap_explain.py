import shap
import joblib
import matplotlib.pyplot as plt
from src.data_preprocessing import load_and_preprocess

def explain(dataset_path, target_column, model_path="models/saved_rf.pkl"):
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(dataset_path, target_column)

    clf = joblib.load(model_path)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("results/shap_summary.png")
    print("SHAP explanation saved at results/shap_summary.png")
