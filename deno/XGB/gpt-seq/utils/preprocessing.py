import pandas as pd
import numpy as np

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

def encode_frequency(seq):
    seq = seq.upper()
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]

def safe_parse_cpp_code(cpp_code_str):
    code = str(cpp_code_str).strip().zfill(20)  # 补齐前导零
    if len(code) != 20 or any(c not in "01" for c in code):
        raise ValueError(f"无效 CPP_Code: {cpp_code_str}")
    return [int(c) for c in code]

def encode_combined_features(seq, cpp_code_str):
    freq_feat = encode_frequency(seq)
    binary_vector = safe_parse_cpp_code(cpp_code_str)
    return np.array(freq_feat + binary_vector)

def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx", binary_col="CPP_Code"):
    train_df = pd.read_excel(train_path, converters={"Sequence": str, binary_col: str})
    test_df = pd.read_excel(test_path, converters={"Sequence": str, binary_col: str})

    X_train = np.array([
        encode_combined_features(row["Sequence"], row[binary_col])
        for _, row in train_df.iterrows()
    ])
    X_test = np.array([
        encode_combined_features(row["Sequence"], row[binary_col])
        for _, row in test_df.iterrows()
    ])
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test