# evaluate.py

import pandas as pd
import numpy as np
import joblib
from utils.evaluation import ClassificationEvaluator
from itertools import product

# 构建k-mer索引
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
KMER_K = 2
ALL_KMERS = [''.join(p) for p in product(AMINO_ACIDS, repeat=KMER_K)]
KMER_INDEX = {kmer: idx for idx, kmer in enumerate(ALL_KMERS)}

def encode_kmer_frequency(seq, k=2):
    seq = seq.upper()
    vec = np.zeros(len(ALL_KMERS))
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer in KMER_INDEX:
            vec[KMER_INDEX[kmer]] += 1
    total = np.sum(vec)
    return vec / total if total > 0 else vec

def load_data_and_model(model_path="models/svm_model.pkl", test_path="data/test.xlsx"):
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_test = np.array([encode_kmer_frequency(seq) for seq in test_df["Sequence"]])
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