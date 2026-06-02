# units/processing_onehot.py

import numpy as np
import pandas as pd

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode(seq, max_len=50):
    """将氨基酸序列进行 One-Hot 编码，不足补X，超长截断"""
    seq = seq.upper()[:max_len].ljust(max_len, "X")
    one_hot = np.zeros((max_len, len(AMINO_ACIDS)), dtype=int)
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX:
            one_hot[i, AA_TO_IDX[aa]] = 1
    return one_hot.flatten()

def load_and_preprocess_onehot(train_path="data/train.xlsx", test_path="data/test.xlsx", max_len=50):
    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_train = train_df["Sequence"].apply(lambda x: one_hot_encode(x, max_len)).tolist()
    X_test = test_df["Sequence"].apply(lambda x: one_hot_encode(x, max_len)).tolist()
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return np.array(X_train), np.array(X_test), y_train, y_test