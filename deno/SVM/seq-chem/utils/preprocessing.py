# units/processing.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# 定义氨基酸理化性质
AA_PROPERTIES = {
    'A': {'hydro': 1.8, 'polar': 8.1, 'mass': 89.1, 'charge': 0},
    'C': {'hydro': 2.5, 'polar': 5.5, 'mass': 121.2, 'charge': 0},
    'D': {'hydro': -3.5, 'polar': 13.0, 'mass': 133.1, 'charge': -1},
    'E': {'hydro': -3.5, 'polar': 12.3, 'mass': 147.1, 'charge': -1},
    'F': {'hydro': 2.8, 'polar': 5.2, 'mass': 165.2, 'charge': 0},
    'G': {'hydro': -0.4, 'polar': 9.0, 'mass': 75.1, 'charge': 0},
    'H': {'hydro': -3.2, 'polar': 10.4, 'mass': 155.2, 'charge': 1},
    'I': {'hydro': 4.5, 'polar': 5.2, 'mass': 131.2, 'charge': 0},
    'K': {'hydro': -3.9, 'polar': 11.3, 'mass': 146.2, 'charge': 1},
    'L': {'hydro': 3.8, 'polar': 4.9, 'mass': 131.2, 'charge': 0},
    'M': {'hydro': 1.9, 'polar': 5.7, 'mass': 149.2, 'charge': 0},
    'N': {'hydro': -3.5, 'polar': 11.6, 'mass': 132.1, 'charge': 0},
    'P': {'hydro': -1.6, 'polar': 8.0, 'mass': 115.1, 'charge': 0},
    'Q': {'hydro': -3.5, 'polar': 10.5, 'mass': 146.1, 'charge': 0},
    'R': {'hydro': -4.5, 'polar': 10.5, 'mass': 174.2, 'charge': 1},
    'S': {'hydro': -0.8, 'polar': 9.2, 'mass': 105.1, 'charge': 0},
    'T': {'hydro': -0.7, 'polar': 8.6, 'mass': 119.1, 'charge': 0},
    'V': {'hydro': 4.2, 'polar': 5.9, 'mass': 117.1, 'charge': 0},
    'W': {'hydro': -0.9, 'polar': 5.4, 'mass': 204.2, 'charge': 0},
    'Y': {'hydro': -1.3, 'polar': 6.2, 'mass': 181.2, 'charge': 0},
}

def encode_physicochemical_features(seq):
    """将氨基酸序列转换为理化性质的统计特征（均值+标准差）"""
    seq = seq.upper()
    props = ['hydro', 'polar', 'mass', 'charge']
    features = []

    for p in props:
        values = [AA_PROPERTIES[aa][p] for aa in seq if aa in AA_PROPERTIES]
        if values:
            features.extend([np.mean(values), np.std(values)])
        else:
            features.extend([0.0, 0.0])  # 空序列或无效字符

    return features  # 最终输出长度为 4 * 2 = 8维


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx"):
    """加载数据并提取理化性质特征"""
    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_train = np.array(train_df["Sequence"].apply(encode_physicochemical_features).tolist())
    X_test = np.array(test_df["Sequence"].apply(encode_physicochemical_features).tolist())
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test


def train_svm_model(X_train, y_train, X_test, y_test,
                   model_path="models/svm_phys_model.pkl", kernel='linear', C=1.0):
    """使用 SVM 训练并保存模型"""
    model = SVC(kernel=kernel, C=C, probability=True)  # 启用概率输出
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
    """用 SVM 预测新数据（理化性质特征）"""
    X_new = np.array([encode_physicochemical_features(seq) for seq in sequences])
    return model.predict(X_new)