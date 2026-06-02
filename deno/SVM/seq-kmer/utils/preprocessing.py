# units/processing.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from itertools import product

# 所有标准氨基酸
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
KMER_K = 2

# 构建所有可能的 k-mer（400个）
ALL_KMERS = [''.join(p) for p in product(AMINO_ACIDS, repeat=KMER_K)]
KMER_INDEX = {kmer: idx for idx, kmer in enumerate(ALL_KMERS)}  # 映射到 index


def encode_kmer_frequency(seq, k=2):
    """提取k-mer频率特征，返回400维向量"""
    seq = seq.upper()
    vec = np.zeros(len(ALL_KMERS))

    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if kmer in KMER_INDEX:
            vec[KMER_INDEX[kmer]] += 1

    total = np.sum(vec)
    return vec / total if total > 0 else vec


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx"):
    """加载数据并提取k-mer频率特征"""
    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_train = np.array(train_df["Sequence"].apply(encode_kmer_frequency).tolist())
    X_test = np.array(test_df["Sequence"].apply(encode_kmer_frequency).tolist())
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test


def train_svm_model(X_train, y_train, X_test, y_test,
                    model_path="models/svm_model.pkl", kernel='linear', C=1.0):
    model = SVC(kernel=kernel, C=C, probability=True)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])
    return model, report


def load_model(model_path="models/svm_kmer2_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    return joblib.load(model_path)


def predict_cpp(model, sequences):
    X_new = np.array([encode_kmer_frequency(seq) for seq in sequences])
    return model.predict(X_new)