import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data
import joblib
import os

def train_model():
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. 训练逻辑回归模型
    print("Training Logistic Regression...")
    log_reg = LogisticRegression(
        max_iter=1000,  # 增加迭代次数以确保收敛
        random_state=42,
        class_weight='balanced'  # 对不平衡的类别进行处理
    )
    log_reg.fit(X_train, y_train)

    # 3. 评估模型
    print("\nModel Evaluation:")
    y_pred = log_reg.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 4. 保存模型
    os.makedirs('models', exist_ok=True)
    joblib.dump(log_reg, 'models/log_reg_model.pkl')
    print("Model saved to models/log_reg_model.pkl")


if __name__ == "__main__":
    train_model()