import pandas as pd
import numpy as np
from sklearn.svm import SVC  # 引入支持向量机（SVM）
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data  # 这里假设有数据加载和预处理的函数
import joblib
import os


def train_model():
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. 训练模型
    print("Training Support Vector Machine (SVM)...")
    svm = SVC(kernel='linear', C=1.0, probability=True)  # 启用概率输出
    svm.fit(X_train, y_train)

    # 3. 评估模型
    print("\nModel Evaluation:")
    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 4. 保存模型
    os.makedirs('models', exist_ok=True)
    joblib.dump(svm, 'models/svm_model.pkl')
    print("Model saved to models/svm_model.pkl")


if __name__ == "__main__":
    train_model()