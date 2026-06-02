# units/processing.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import ast  # 处理列表字符串形式


def parse_cpp_code(code):
    """
    将 CPP_Code 列中的内容解析为 20维 0/1 数组
    支持：字符串 '101010...' 或 '[1, 0, 1, ...]'
    """
    if isinstance(code, list):
        return np.array(code, dtype=int)
    if isinstance(code, str):
        code = code.strip()
        if code.startswith("["):
            return np.array(ast.literal_eval(code), dtype=int)
        else:
            return np.array([int(c) for c in code], dtype=int)
    raise ValueError(f"无法解析 CPP_Code: {code}")


def load_and_preprocess_data(train_path="data/train.xlsx", test_path="data/test.xlsx"):
    train_df = pd.read_excel(train_path, converters={"CPP_Code": str})
    test_df = pd.read_excel(test_path, converters={"CPP_Code": str})

    X_train = np.array(train_df["CPP_Code"].apply(parse_cpp_code).tolist())
    X_test = np.array(test_df["CPP_Code"].apply(parse_cpp_code).tolist())
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test


def train_svm_model(X_train, y_train, X_test, y_test,
                    model_path="models/svm_cppcode_model.pkl", kernel='linear', C=1.0):
    model = SVC(kernel=kernel, C=C, probability=True)
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])
    return model, report


def load_model(model_path="models/svm_cppcode_model.pkl"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    return joblib.load(model_path)


def predict_cpp(model, cpp_code_list):
    """cpp_code_list: List of 20位字符串或数组"""
    X_new = np.array([parse_cpp_code(code) for code in cpp_code_list])
    return model.predict(X_new)