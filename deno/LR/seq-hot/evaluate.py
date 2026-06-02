# evaluate.py

import os
import joblib
import pandas as pd
from utils.preprocessing import one_hot_encode_sequence
from utils.evaluation import ClassificationEvaluator

def load_test_data(filepath="data/test.xlsx", max_len=50):
    df = pd.read_excel(filepath, converters={"Sequence": str})
    X = df["Sequence"].apply(lambda x: one_hot_encode_sequence(x, max_len)).tolist()
    y = df["Label"].values
    return X, y

def evaluate_model():
    print("Loading model and test data...")
    model = joblib.load("models/rf_model.pkl")
    X_test, y_test = load_test_data()

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    evaluator = ClassificationEvaluator(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"],
        output_dir="results"
    )

    print("\n" + "=" * 50)
    print(" Quantitative Metrics ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    print("\nGenerating visualizations...")
    evaluator.plot_all()

if __name__ == "__main__":
    evaluate_model()