import joblib
import pandas as pd
import numpy as np
from utils.evaluation import ClassificationEvaluator
from sklearn.preprocessing import LabelEncoder

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def encode_frequency(seq):
    """氨基酸频率特征：20维"""
    seq = seq.upper()
    total = len(seq)
    return [seq.count(aa) / total if total > 0 else 0 for aa in AMINO_ACIDS]


def load_data_and_model(model_path="models/xgb_model.pkl",
                        test_path="data/test.xlsx"):
    # 加载数据
    test_df = pd.read_excel(test_path, converters={"Sequence": str})
    X_test = np.array([encode_frequency(seq) for seq in test_df["Sequence"]])
    y_test = test_df["Label"].values

    # 加载模型
    model = joblib.load(model_path)

    # 预测标签
    y_pred = model.predict(X_test)

    # 预测概率（如果支持）
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        # 如果返回一维，自动转成二维概率（避免 IndexError）
        if y_proba.ndim == 1:
            y_proba = np.vstack([1 - y_proba, y_proba]).T

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
    print(" Evaluation Metrics ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    print("\nGenerating visualizations...")
    evaluator.plot_all()