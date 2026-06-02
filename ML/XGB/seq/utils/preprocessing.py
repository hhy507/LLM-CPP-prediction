import pandas as pd
import numpy as np

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def encode_frequency(seq):
    """将氨基酸序列编码为频率向量（20维）"""
    seq = seq.upper()
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]


def load_data(train_path="data/train.xlsx", test_path="data/test.xlsx"):
    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_train = np.array([encode_frequency(seq) for seq in train_df["Sequence"]])
    X_test = np.array([encode_frequency(seq) for seq in test_df["Sequence"]])
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test