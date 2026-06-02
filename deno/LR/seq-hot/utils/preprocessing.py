# utils/processing.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

def one_hot_encode_sequence(seq, max_len=50):
    """将氨基酸序列转换为 One-Hot 编码"""
    seq = seq.upper()
    encoding = np.zeros((max_len, len(AMINO_ACIDS)), dtype=int)

    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_INDEX:
            encoding[i, AA_TO_INDEX[aa]] = 1
    return encoding.flatten()

def load_and_preprocess_data(train_path="data/train.xlsx", max_len=50, test_size=0.2):
    df = pd.read_excel(train_path, converters={"Sequence": str})
    X = df["Sequence"].apply(lambda s: one_hot_encode_sequence(s, max_len)).tolist()
    y = df["Label"].values
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)