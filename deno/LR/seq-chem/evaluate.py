# evaluate.py

import os
import joblib
import pandas as pd
from utils.evaluation import ClassificationEvaluator
from utils.preprocessing import encode_physicochemical_features  # 自定义特征编码器

def load_test_data(filepath):
    """加载并处理测试数据"""
    df = pd.read_excel(filepath, converters={"Sequence": str})
    X = df["Sequence"].apply(encode_physicochemical_features).tolist()
    y = df["Label"].values
    return X, y

def evaluate_model(model_path, test_data_path, output_dir="results"):
    # 1. 加载模型
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)

    # 2. 加载测试数据
    print(f"Loading test data from {test_data_path}...")
    X_test, y_true = load_test_data(test_data_path)

    # 3. 模型预测
    print("Predicting...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # 4. 初始化评估器
    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"],
        output_dir=output_dir
    )

    # 5. 打印并保存指标
    print("\nEvaluation Report:")
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    # 6. 生成可视化图像
    print("\nGenerating plots...")
    evaluator.plot_all()

if __name__ == "__main__":
    MODEL_PATH = "models/rf_model.pkl"
    TEST_DATA_PATH = "data/test.xlsx"
    OUTPUT_DIR = "results"

    evaluate_model(MODEL_PATH, TEST_DATA_PATH, OUTPUT_DIR)