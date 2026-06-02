import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data


def train_model():
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. 训练模型
    print("Training XGBoost Classifier...")
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # 平衡类别
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # 3. 评估模型
    print("\nModel Evaluation:")
    y_pred = xgb_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 4. 保存模型
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgb_combined_model.pkl')
    print("Model saved to models/xgb_combined_model.pkl")


if __name__ == "__main__":
    train_model()