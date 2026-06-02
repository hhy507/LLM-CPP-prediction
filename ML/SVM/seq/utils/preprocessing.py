# units/processing.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def encode_frequency(seq):
    """氨基酸频率特征：20维"""
    seq = seq.upper()
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx"):
    """加载数据并提取氨基酸频率特征"""
    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_train = np.array(train_df["Sequence"].apply(encode_frequency).tolist())
    X_test = np.array(test_df["Sequence"].apply(encode_frequency).tolist())
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test


def train_svm_model(X_train, y_train, X_test, y_test,
                   model_path="models/svm_model.pkl", kernel='linear', C=1.0):
    """使用 SVM 训练并保存模型"""
    model = SVC(kernel=kernel, C=C)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])
    return model, report


def load_model(model_path="models/svm_model.pkl"):
    """从本地加载模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    return joblib.load(model_path)


def predict_cpp(model, sequences):
    """用 SVM 预测新数据（氨基酸频率特征）"""
    X_new = np.array([encode_frequency(seq) for seq in sequences])
    return model.predict(X_new)