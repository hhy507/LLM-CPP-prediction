# evaluate.py

import os
import pandas as pd
import numpy as np
import joblib
from utils.evaluation import ClassificationEvaluator
import ast

def parse_cpp_code(code):
    if isinstance(code, list):
        return np.array(code, dtype=int)
    if isinstance(code, str):
        code = code.strip()
        if code.startswith("["):
            return np.array(ast.literal_eval(code), dtype=int)
        else:
            return np.array([int(c) for c in code], dtype=int)
    raise ValueError(f"无法解析 CPP_Code: {code}")


def load_data_and_model(model_path="models/svm_model.pkl", test_path="data/test.xlsx"):
    test_df = pd.read_excel(test_path, converters={"CPP_Code": str})

    X_test = np.array([parse_cpp_code(code) for code in test_df["CPP_Code"]])
    y_test = test_df["Label"].values

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

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