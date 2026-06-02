import joblib
import pandas as pd
import numpy as np
from utils.evaluation import ClassificationEvaluator
from sklearn.preprocessing import StandardScaler

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def encode_frequency(seq):
    """氨基酸频率特征：20维"""
    seq = seq.upper()
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]


def encode_combined_features(seq, binary_features):
    """合并氨基酸频率 + 自定义20维01编码"""
    freq_feat = encode_frequency(seq)
    return np.array(freq_feat + list(binary_features))  # 合并为一个特征向量


def load_data_and_model(model_path="models/lr_combined_model.pkl",
                        test_path="data/test.xlsx", binary_cols=None):
    if binary_cols is None:
        binary_cols = ["CPP_Code"]  # 自定义列名 "CPP_Code" 为 20维特征列

    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    # 提取自定义 20维特征并合并
    X_test = np.array([
        encode_combined_features(row["Sequence"], row[binary_cols].values)
        for _, row in test_df.iterrows()
    ])
    y_test = test_df["Label"].values

    # 加载训练好的模型（逻辑回归模型）
    model = joblib.load(model_path)

    # 特征标准化
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

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