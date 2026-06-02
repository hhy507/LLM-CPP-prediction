import joblib
import pandas as pd
import numpy as np
from utils.evaluation import ClassificationEvaluator

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def encode_frequency(seq):
    seq = seq.upper()
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]

def safe_parse_cpp_code(cpp_code_str):
    code = str(cpp_code_str).strip().zfill(20)
    if len(code) != 20 or any(c not in "01" for c in code):
        raise ValueError(f"无效 CPP_Code: {cpp_code_str}")
    return [int(c) for c in code]

def encode_combined_features(seq, cpp_code_str):
    freq_feat = encode_frequency(seq)
    binary_vector = safe_parse_cpp_code(cpp_code_str)
    return np.array(freq_feat + binary_vector)

def load_data_and_model(model_path="models/xgboost_model.pkl", test_path="data/test.xlsx", binary_col="CPP_Code"):
    test_df = pd.read_excel(test_path, converters={"Sequence": str, binary_col: str})

    X_test = np.array([
        encode_combined_features(row["Sequence"], row[binary_col])
        for _, row in test_df.iterrows()
    ])
    y_test = test_df["Label"].values

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

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