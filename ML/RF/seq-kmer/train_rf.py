import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data
import joblib
import os


def train_model():
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. 训练模型
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        oob_score = True  # 启用袋外估计
    )
    rf.fit(X_train, y_train)

    # 3. 评估模型
    print("\nModel Evaluation:")
    y_pred = rf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 4. 保存模型
    os.makedirs('models', exist_ok=True)
    joblib.dump(rf, 'models/rf_model.pkl')
    print("Model saved to models/rf_model.pkl")


if __name__ == "__main__":
    train_model()