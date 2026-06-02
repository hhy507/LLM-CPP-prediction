import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def generate_kmers(sequence: str, k: int = 2):
    """生成序列的所有k-mer"""
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

def build_kmer_vocab(k: int = 2):
    """构建所有可能的k-mer组合"""
    return [''.join(p) for p in product(AMINO_ACIDS, repeat=k)]

def extract_kmer_features(sequences, k: int = 2):
    """提取所有序列的k-mer特征向量"""
    vocab = build_kmer_vocab(k)
    kmer_index = {kmer: idx for idx, kmer in enumerate(vocab)}
    features = []

    for seq in sequences:
        seq = seq.upper()
        vec = np.zeros(len(vocab), dtype=float)
        for kmer in generate_kmers(seq, k):
            if kmer in kmer_index:
                vec[kmer_index[kmer]] += 1
        if vec.sum() > 0:
            vec /= vec.sum()
        features.append(vec)

    return pd.DataFrame(features, columns=vocab)

def load_and_preprocess_data(train_file="data/train.xlsx", k: int = 2, test_size=0.2, random_state=42):
    """
    加载训练集数据，提取k-mer特征，并划分训练和测试集
    """
    df = pd.read_excel(train_file, converters={"Sequence": str})
    sequences = df["Sequence"].astype(str).tolist()
    labels = df["Label"].values

    X = extract_kmer_features(sequences, k=k)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    return X_train, X_test, y_train, y_test