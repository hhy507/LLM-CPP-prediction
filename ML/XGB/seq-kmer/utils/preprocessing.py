# units/processing_kmer.py

import itertools
import numpy as np
import pandas as pd
from collections import Counter

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def generate_kmer_vocab(k=2):
    """生成所有可能的k-mer组合"""
    return [''.join(p) for p in itertools.product(AMINO_ACIDS, repeat=k)]

def kmer_count(seq, k=2, vocab=None):
    """将序列转换为k-mer频率向量"""
    seq = seq.upper()
    kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
    counts = Counter(kmers)
    if vocab is None:
        vocab = generate_kmer_vocab(k)
    return [counts[kmer] for kmer in vocab]

def load_and_preprocess_kmer(train_path="data/train.xlsx", test_path="data/test.xlsx", k=2):
    vocab = generate_kmer_vocab(k)

    train_df = pd.read_excel(train_path, converters={"Sequence": str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str})

    X_train = train_df["Sequence"].apply(lambda seq: kmer_count(seq, k=k, vocab=vocab)).tolist()
    X_test = test_df["Sequence"].apply(lambda seq: kmer_count(seq, k=k, vocab=vocab)).tolist()

    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return np.array(X_train), np.array(X_test), y_train, y_test, vocab