# evaluate.py

import pandas as pd
import numpy as np
import joblib
from utils.evaluation import ClassificationEvaluator

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode_sequence(seq, max_length=20):
    seq = seq.upper()[:max_length].ljust(max_length, 'X')
    vec = np.zeros((max_length, len(AMINO_ACIDS)))
    for i, aa in enumerate(seq):
        if aa in AA_INDEX:
            vec[i][AA_INDEX[aa]] = 1
    return vec.flatten()

def load_data_and_model(model_path="models/svm_model.pkl", test_path="data/test.xlsx", max_length=20):
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_test = np.array([one_hot_encode_sequence(seq, max_length) for seq in test_df["Sequence"]])
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