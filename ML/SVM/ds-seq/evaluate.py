# evaluate.py

import numpy as np
import pandas as pd
import joblib
from utils.evaluation import ClassificationEvaluator
from utils.preprocessing import encode_combined_features, parse_cpp_code
import ast

def load_data_and_model(test_path="data/test.xlsx",
                        model_path="models/svm_model.pkl",
                        scaler_path="models/scaler.pkl"):
    test_df = pd.read_excel(test_path, converters={"Sequence": str, "CPP_Code": str})
    test_df["CPP_Code"] = test_df["CPP_Code"].apply(lambda x: str(x).strip().zfill(20))

    X_test = np.array([
        encode_combined_features(row["Sequence"], parse_cpp_code(row["CPP_Code"]))
        for _, row in test_df.iterrows()
    ])
    y_test = test_df["Label"].values

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled) if hasattr(model, "predict_proba") else None

    return y_test, y_pred, y_proba

if __name__ == "__main__":
    y_true, y_pred, y_proba = load_data_and_model()

    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]
    )

    print("\n" + "=" * 50)
    print(" Quantitative Metrics ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    print("\nGenerating visualizations...")
    evaluator.plot_all()