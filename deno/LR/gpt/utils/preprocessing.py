import pandas as pd
import numpy as np

def load_and_preprocess_data():
    """加载数据并确保CPP_Code保留前导零"""
    # ✅ 关键修改：使用converters强制保持字符串格式
    train_df = pd.read_excel(
        "data/train.xlsx",
        converters={"CPP_Code": str}  # 防止前导零丢失
    )
    test_df = pd.read_excel(
        "data/test.xlsx",
        converters={"CPP_Code": str}  # 同样处理测试集
    )

    # 编码转换函数（安全处理）
    def encode_cpp(code, fixed_length=20):
        return [int(i) for i in str(code).zfill(fixed_length)]

    # 应用处理
    X_train = np.array(train_df["CPP_Code"].apply(encode_cpp).tolist())
    X_test = np.array(test_df["CPP_Code"].apply(encode_cpp).tolist())
    y_train = train_df["Label"].values
    y_test = test_df["Label"].values

    return X_train, X_test, y_train, y_test