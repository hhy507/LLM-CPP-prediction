import joblib
import pandas as pd
from utils.evaluation import ClassificationEvaluator


def load_data_and_model():
    """加载测试数据和训练好的模型"""
    # 1. 加载测试数据
    test_df = pd.read_excel("data/test.xlsx",converters={"CPP_Code": str} )
    X_test = test_df["CPP_Code"].apply(lambda x: [int(i) for i in x.zfill(20)]).tolist()
    y_test = test_df["Label"].values

    # 2. 加载模型
    model = joblib.load("models/rf_model.pkl")

    # 3. 获取预测结果
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    return y_test, y_pred, y_proba


if __name__ == "__main__":
    y_true, y_pred, y_proba = load_data_and_model()

    # 初始化评估器
    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]  # 修改为你的类别名称
    )

    # 打印数值报告
    print("\n" + "=" * 50)
    print(" Quantitative Metrics ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    # 绘制所有可视化
    print("\nGenerating visualizations...")
    evaluator.plot_all()