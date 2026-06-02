# utils/processing_freq_lr.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def encode_frequency(seq):
    seq = seq.upper()
    length = len(seq)
    if length == 0:
        return np.zeros(len(AMINO_ACIDS))
    return np.array([seq.count(aa) / length for aa in AMINO_ACIDS])

def preprocess_and_split(data_path="data/train.xlsx", test_size=0.2, random_state=42):
    df = pd.read_excel(data_path, converters={"Sequence": str})
    df["FreqVector"] = df["Sequence"].apply(encode_frequency)

    X = np.stack(df["FreqVector"].values)
    y = df["Label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/freq_lr_scaler.pkl")

    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)

def train_and_save_lr(X_train, y_train, model_path="models/freq_logistic_regression.pkl"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)