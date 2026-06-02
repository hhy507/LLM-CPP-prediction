# evaluate_kmer.py

import joblib
import numpy as np
import pandas as pd
from utils.preprocessing import kmer_count, generate_kmer_vocab
from utils.evaluation import ClassificationEvaluator

def evaluate_kmer(model_path="models/xgb_kmer2_model.pkl", test_path="data/test.xlsx", k=2):
    vocab = generate_kmer_vocab(k)
    test_df = pd.read_excel(test_path, converters={"Sequence": str})
    X_test = np.array([kmer_count(seq, k=k, vocab=vocab) for seq in test_df["Sequence"]])
    y_test = test_df["Label"].values

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    evaluator = ClassificationEvaluator(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]
    )

    print("\n" + "=" * 50)
    print(f" 2-mer XGBoost Evaluation ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    evaluator.plot_all()

if __name__ == "__main__":
    evaluate_kmer(k=2)