# evaluate_onehot.py

import joblib
import numpy as np
import pandas as pd
from utils.preprocessing import one_hot_encode
from utils.evaluation import ClassificationEvaluator

def evaluate_onehot(model_path="models/xgb_onehot_model.pkl", test_path="data/test.xlsx", max_len=50):
    test_df = pd.read_excel(test_path, converters={"Sequence": str})
    X_test = np.array([one_hot_encode(seq, max_len) for seq in test_df["Sequence"]])
    y_test = test_df["Label"].values

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)

    evaluator = ClassificationEvaluator(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]
    )

    print("\n" + "=" * 50)
    print(" One-Hot XGBoost Evaluation ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    evaluator.plot_all()

if __name__ == "__main__":
    evaluate_onehot()