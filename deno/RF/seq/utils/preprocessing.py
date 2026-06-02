# units/processing_sequence.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 所有标准氨基酸
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def encode_sequence(seq):
    """将氨基酸序列编码为频率向量"""
    seq = seq.upper()
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx"):
    """加载数据并将 Sequence 转换为氨基酸频率特征"""
    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    X_train = np.array(train_df["Sequence"].apply(encode_sequence).tolist())
    X_test = np.array(test_df["Sequence"].apply(encode_sequence).tolist())
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test


def train_rf_model(X_train, y_train, X_test, y_test, model_path="rf_seq_model.pkl",
                   n_estimators=100, random_state=42):
    """训练并保存随机森林模型（基于氨基酸频率）"""
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)

    # 保存模型
    joblib.dump(model, model_path)

    # 预测与报告
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])
    return model, report


def load_model(model_path="rf_seq_model.pkl"):
    """从本地加载保存的模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    return joblib.load(model_path)


def predict_cpp(model, sequences):
    """使用模型预测新氨基酸序列是否为 CPP"""
    def encode_sequence(seq):
        seq = seq.upper()
        return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]

    X_new = np.array([encode_sequence(seq) for seq in sequences])
    return model.predict(X_new)