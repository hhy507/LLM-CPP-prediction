import joblib
import pandas as pd
from utils.evaluation import ClassificationEvaluator


def load_data_and_model():
    """加载测试数据和训练好的模型"""
    # 1. 加载测试数据
    # 这里是 encode_sequence 的正确版本
    AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

    def encode_sequence(seq):
        """将氨基酸序列转换为频率向量"""
        seq = seq.upper()
        return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]
    test_df = pd.read_excel("data/test.xlsx",converters={"Sequence": str} )
    X_test = test_df["Sequence"].apply(encode_sequence).tolist()
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