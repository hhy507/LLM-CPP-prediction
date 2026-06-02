# evaluate_freq_lr.py

import pandas as pd
import numpy as np
import joblib
from utils.evaluation import ClassificationEvaluator
from utils.preprocessing import encode_frequency

def load_and_predict(model_path="models/freq_logistic_regression.pkl",
                     test_path="data/test.xlsx",
                     scaler_path="models/freq_lr_scaler.pkl"):
    df = pd.read_excel(test_path, converters={"Sequence": str})
    df["FreqVector"] = df["Sequence"].apply(encode_frequency)
    X = np.stack(df["FreqVector"].values)
    y_true = df["Label"].values

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    model = joblib.load(model_path)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    return y_true, y_pred, y_proba

if __name__ == "__main__":
    y_true, y_pred, y_proba = load_and_predict()

    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]
    )

    print("\n" + "=" * 50)
    print(" Logistic Regression (Frequency Features) ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<20}: {value:.4f}")

    print("\nGenerating visualizations...")
    evaluator.plot_all()