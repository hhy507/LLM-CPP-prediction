import joblib
import numpy as np
import pandas as pd
from utils.evaluation import ClassificationEvaluator
from utils.preprocessing import encode_physicochemical  # 导入处理函数

def load_data_and_model(model_path="models/xgb_phys_model.pkl", test_path="data/test.xlsx"):
    """加载数据与模型并进行预测"""
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    # 提取理化性质特征
    X_test = np.array([encode_physicochemical(seq) for seq in test_df["Sequence"]])
    y_test = test_df["Label"].values

    # 加载模型
    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    # 获取预测概率
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if y_proba.ndim == 1:
            y_proba = np.vstack([1 - y_proba, y_proba]).T

    return y_test, y_pred, y_proba

if __name__ == "__main__":
    y_true, y_pred, y_proba = load_data_and_model()

    # 评估器
    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]
    )

    # 输出评估报告
    print("\n" + "=" * 50)
    print(" Physicochemical Evaluation ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    print("\nGenerating visualizations...")
    evaluator.plot_all()