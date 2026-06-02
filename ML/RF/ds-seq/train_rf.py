# train_model_40_features.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# 定义氨基酸
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def encode_frequency(seq):
    """计算20维氨基酸频率"""
    seq = seq.upper()
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]

def encode_combined_features(seq, binary_features):
    """合并频率特征 + 规则特征"""
    freq_feat = encode_frequency(seq)
    return np.array(freq_feat + list(binary_features))

def load_and_preprocess_data(data_path="data/train.xlsx"):
    """读取数据并处理成40维特征"""
    binary_cols = [f"Rule_{i}" for i in range(1, 21)]

    df = pd.read_excel(data_path, converters={"Sequence": str})

    X = np.array([
        encode_combined_features(row["Sequence"], row[binary_cols].values)
        for _, row in df.iterrows()
    ])
    y = df["Label"].values

    return X, y

def train_and_save_model(X, y, model_path="models/rf_40features_model.pkl"):
    """训练随机森林并保存"""
    # 分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"训练集样本数: {X_train.shape[0]}")
    print(f"测试集样本数: {X_test.shape[0]}")

    # 训练模型
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        oob_score=True
    )
    clf.fit(X_train, y_train)

    # 保存模型
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"✅ 模型已保存到: {model_path}")

    # 评估
    print("\n📊 训练集表现：")
    y_train_pred = clf.predict(X_train)
    print(classification_report(y_train, y_train_pred))

    print("\n📈 测试集表现：")
    y_test_pred = clf.predict(X_test)
    print(classification_report(y_test, y_test_pred))

def main():
    X, y = load_and_preprocess_data(data_path="data/train.xlsx")  # 注意你的路径！
    train_and_save_model(X, y, model_path="models/rf_40features_model.pkl")

if __name__ == "__main__":
    main()