import os
import joblib
import pandas as pd
from collections import Counter
from itertools import product
from utils.evaluation import ClassificationEvaluator  # 你自定义的评估器

# ========== 特征工程：k-mer 提取 ==========
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def generate_kmers(sequence, k=2):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def get_all_possible_kmers(k=2):
    return [''.join(p) for p in product(AMINO_ACIDS, repeat=k)]

def encode_kmer_frequency(sequences, k=2):
    """将序列列表编码成 k-mer 频率向量"""
    all_kmers = get_all_possible_kmers(k)
    kmer_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    num_features = len(all_kmers)

    features = []
    for seq in sequences:
        vec = [0] * num_features
        kmers = generate_kmers(seq.upper(), k)
        counts = Counter(kmers)
        total = sum(counts.values())
        for kmer, count in counts.items():
            if kmer in kmer_index:
                vec[kmer_index[kmer]] = count / total
        features.append(vec)
    return features


# ========== 数据与模型加载 ==========
def load_data_and_model(test_path="data/test.xlsx", model_path="models/rf_model.pkl", k=2):
    """加载测试数据和训练好的模型"""
    # 1. 加载测试数据
    test_df = pd.read_excel(test_path, converters={"Sequence": str})
    X_test = encode_kmer_frequency(test_df["Sequence"].tolist(), k=k)
    y_test = test_df["Label"].values

    # 2. 加载模型
    model = joblib.load(model_path)

    # 3. 预测
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    return y_test, y_pred, y_proba


# ========== 主程序 ==========
if __name__ == "__main__":
    # 参数：k-mer 的 k 值必须和训练时一致！
    k = 2
    model_path = "models/rf_model.pkl"
    test_path = "data/test.xlsx"

    y_true, y_pred, y_proba = load_data_and_model(test_path, model_path, k)

    # 初始化评估器
    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Negative", "Positive"]  # 根据你任务的类别来定
    )

    # 打印数值评估报告
    print("\n" + "=" * 50)
    print(" Quantitative Metrics ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    # 生成可视化图
    print("\nGenerating visualizations...")
    evaluator.plot_all()