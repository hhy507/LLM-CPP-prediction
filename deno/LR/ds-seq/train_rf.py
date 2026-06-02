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

    # 2. 训练模型
    print("Training Logistic Regression...")
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,  # 设置最大迭代次数
        class_weight='balanced',  # 处理不平衡类
        solver='liblinear'  # 可以选择其他求解器，如 'liblinear', 'saga' 等
    )
    lr.fit(X_train, y_train)

    # 3. 评估模型
    print("\nModel Evaluation:")
    y_pred = lr.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 4. 保存模型
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr, 'models/lr_combined_model.pkl')
    print("Model saved to models/lr_combined_model.pkl")


if __name__ == "__main__":
    train_model()