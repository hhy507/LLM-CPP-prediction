# units/processing.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 氨基酸种类（按顺序作为 one-hot 的维度）
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def one_hot_encode_sequence(seq, max_length=20):
    """
    对氨基酸序列进行one-hot编码，保留位置
    返回: 长度 max_length × 20 的向量
    """
    seq = seq.upper()[:max_length].ljust(max_length, 'X')  # 截断+填充
    vec = np.zeros((max_length, len(AMINO_ACIDS)))

    for i, aa in enumerate(seq):
        if aa in AA_INDEX:
            vec[i][AA_INDEX[aa]] = 1
    return vec.flatten()  # 展平为一维向量


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx", max_length=20):
    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_train = np.array(train_df["Sequence"].apply(lambda x: one_hot_encode_sequence(x, max_length)).tolist())
    X_test = np.array(test_df["Sequence"].apply(lambda x: one_hot_encode_sequence(x, max_length)).tolist())
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test


def train_rf_model(X_train, y_train, X_test, y_test, model_path="models/rf_onehot_model.pkl",
                   n_estimators=100, random_state=42):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])
    return model, report


def load_model(model_path="models/rf_onehot_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    return joblib.load(model_path)


def predict_cpp(model, sequences, max_length=20):
    X_new = np.array([one_hot_encode_sequence(seq, max_length) for seq in sequences])
    return model.predict(X_new)