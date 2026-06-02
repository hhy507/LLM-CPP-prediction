# utils/processing.py

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import os
import ast

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def encode_frequency(sequence):
    """计算氨基酸序列频率（20维）"""
    sequence = sequence.upper()
    total = len(sequence)
    return [sequence.count(aa) / total if total > 0 else 0 for aa in AMINO_ACIDS]

def parse_cpp_code(code):
    """将CPP_Code字段转为长度为20的np.array"""
    if isinstance(code, list):
        return np.array(code, dtype=int)
    if isinstance(code, str):
        code = code.strip()
        if code.startswith("["):
            return np.array(ast.literal_eval(code), dtype=int)
        else:
            return np.array([int(c) for c in code.zfill(20)], dtype=int)
    raise ValueError(f"无法解析 CPP_Code: {code}")

def encode_combined_features(sequence, binary_features):
    """将频率特征和01编码合并为一个特征向量"""
    freq = encode_frequency(sequence)
    if len(binary_features) != 20:
        raise ValueError("Binary feature vector must be 20-dimensional.")
    return np.array(freq + list(binary_features))

def load_and_preprocess_data(train_path="data/train.xlsx",
                              test_path="data/test.xlsx",
                              sequence_col="Sequence",
                              binary_col="CPP_Code"):
    """读取Excel并生成训练和测试数据"""
    train_df = pd.read_excel(train_path, converters={sequence_col: str, binary_col: str})
    test_df = pd.read_excel(test_path, converters={sequence_col: str, binary_col: str})

    for df in [train_df, test_df]:
        df[binary_col] = df[binary_col].apply(lambda x: str(x).strip().zfill(20))

    X_train = np.array([
        encode_combined_features(row[sequence_col], parse_cpp_code(row[binary_col]))
        for _, row in train_df.iterrows()
    ])
    X_test = np.array([
        encode_combined_features(row[sequence_col], parse_cpp_code(row[binary_col]))
        for _, row in test_df.iterrows()
    ])

    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test

def train_and_save_model(X_train, y_train, model_path="models/svm_model.pkl",
                         scaler_path="models/scaler.pkl"):
    """训练 SVM 模型并保存模型与标准化器"""
    os.makedirs("models", exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    model.fit(X_scaled, y_train)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler