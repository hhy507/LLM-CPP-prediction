import os
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data  # 导入处理函数

def train_xgb_model(X_train, y_train, X_test, y_test, model_path="models/xgb_phys_model.pkl"):
    """训练并保存 XGBoost 模型"""
    model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="mlogloss")
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])

    return model, report

if __name__ == "__main__":
    # 加载数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 训练模型
    model, report = train_xgb_model(X_train, y_train, X_test, y_test)

    # 输出分类报告
    print("\n分类报告：")
    print(report)