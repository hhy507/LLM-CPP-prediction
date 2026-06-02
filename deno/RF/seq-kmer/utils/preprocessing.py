# units/processing_kmer.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from collections import Counter
from itertools import product


def generate_kmers(sequence, k=2):
    """滑窗提取所有 k-mer 子串"""
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]


def get_all_possible_kmers(k=2, alphabet='ACDEFGHIKLMNPQRSTVWY'):
    """生成所有可能的 k-mer 组合"""
    return [''.join(p) for p in product(alphabet, repeat=k)]


def encode_kmer_frequency(sequences, k=2):
    """将氨基酸序列转换为 k-mer 频率向量"""
    all_kmers = get_all_possible_kmers(k)
    kmer_index = {kmer: idx for idx, kmer in enumerate(all_kmers)}
    num_features = len(all_kmers)

    features = []
    for seq in sequences:
        vec = np.zeros(num_features)
        kmers = generate_kmers(seq.upper(), k)
        counts = Counter(kmers)
        total = sum(counts.values())

        for kmer, count in counts.items():
            if kmer in kmer_index:
                vec[kmer_index[kmer]] = count / total  # 使用频率
        features.append(vec)

    return np.array(features)


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx", k=2):
    """加载训练和测试数据并提取 k-mer 频率特征"""
    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    sequences_train = train_df["Sequence"].astype(str).tolist()
    sequences_test = test_df["Sequence"].astype(str).tolist()

    X_train = encode_kmer_frequency(sequences_train, k=k)
    X_test = encode_kmer_frequency(sequences_test, k=k)

    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test


def train_rf_model(X_train, y_train, X_test, y_test,
                   model_path="models/rf_kmer_model.pkl",
                   n_estimators=100, random_state=42):
    """训练并保存随机森林模型"""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])

    return model, report


def load_model(model_path="models/rf_kmer_model.pkl"):
    """加载已保存的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    return joblib.load(model_path)


def predict_cpp_kmer(model, sequences, k=2):
    """使用模型预测新的氨基酸序列是否为 CPP"""
    X_new = encode_kmer_frequency(sequences, k=k)
    return model.predict(X_new)