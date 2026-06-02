# processing.py

import numpy as np
import pandas as pd

# ========== 理化性质字典 ==========
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


def encode_physicochemical_features(sequence):
    """将氨基酸序列转换为理化性质的统计特征（均值+标准差）"""
    sequence = sequence.upper()
    props = ['hydro', 'polar', 'mass', 'charge']
    features = []

    for p in props:
        values = [AA_PROPERTIES.get(aa, {'hydro': 0, 'polar': 0, 'mass': 0, 'charge': 0})[p] for aa in sequence]
        features.extend([np.mean(values), np.std(values)])  # 使用均值和标准差

    return features


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx"):
    """加载序列数据并提取理化性质特征"""
    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    # 处理训练和测试数据
    X_train = np.array([encode_physicochemical_features(seq) for seq in train_df["Sequence"]])
    X_test = np.array([encode_physicochemical_features(seq) for seq in test_df["Sequence"]])

    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test