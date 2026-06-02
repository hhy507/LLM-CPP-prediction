# evaluate.py

import joblib
import pandas as pd
import numpy as np
from utils.evaluation import ClassificationEvaluator

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def encode_frequency(seq):
    """氨基酸频率特征：20维"""
    seq = seq.upper()
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]


def load_data_and_model(model_path="models/svm_model.pkl", test_path="data/test.xlsx"):
    """加载数据和模型进行预测"""
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    # 提取氨基酸频率特征
    X_test = np.array([encode_frequency(seq) for seq in test_df["Sequence"]])
    y_test = test_df["Label"].values

    # 加载训练好的 SVM 模型
    model = joblib.load(model_path)

    # 获取预测结果和预测概率
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)  # 获取预测概率

    return y_test, y_pred, y_proba


if __name__ == "__main__":
    y_true, y_pred, y_proba = load_data_and_model()

    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,  # 传递预测的概率值
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