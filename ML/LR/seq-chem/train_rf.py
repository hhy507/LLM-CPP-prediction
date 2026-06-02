# train.py

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data


def train_logistic_regression(X_train, y_train, X_test, y_test, model_path="models/logreg_model.pkl"):
    """训练并保存逻辑回归模型"""
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)

    # 评估模型
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])

    # 保存模型
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model, report


def main():
    # 加载数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 训练逻辑回归模型并评估
    model, report = train_logistic_regression(X_train, y_train, X_test, y_test)

    # 打印评估结果
    print("\nModel Evaluation Report:")
    print(report)


if __name__ == "__main__":
    main()