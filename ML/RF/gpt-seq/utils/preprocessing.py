# units/processing.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx", binary_cols=None):
    if binary_cols is None:
        binary_cols = ["CPP_Code"]  # 默认列名为 "CPP_Code"，你可以修改这个

    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    # 20维自定义二进制特征（从 "CPP_Code" 提取）
    X_train = np.array([
        encode_combined_features(row["Sequence"], row[binary_cols].values)
        for _, row in train_df.iterrows()
    ])
    X_test = np.array([
        encode_combined_features(row["Sequence"], row[binary_cols].values)
        for _, row in test_df.iterrows()
    ])

    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test


def train_rf_model(X_train, y_train, X_test, y_test,
                   model_path="models/rf_combined_model.pkl",
                   n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])
    return model, report


def load_model(model_path="models/rf_combined_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    return joblib.load(model_path)


def predict_cpp(model, sequences, binary_features_list):
    """
    sequences: list of amino acid sequences
    binary_features_list: list of 20-dimensional binary feature vectors (for each sequence)
    """
    X_new = np.array([
        encode_combined_features(seq, binary)
        for seq, binary in zip(sequences, binary_features_list)
    ])
    return model.predict(X_new)