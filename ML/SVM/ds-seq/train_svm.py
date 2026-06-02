import os
from utils.preprocessing import load_and_preprocess_data, train_and_save_model
from sklearn.metrics import classification_report

def train_model():
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 2. 训练模型并保存（含标准化器）
    print("Training Support Vector Machine (SVM)...")
    model, scaler = train_and_save_model(X_train, y_train)

    # 3. 标准化测试集并评估
    X_test_scaled = scaler.transform(X_test)
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))

    print("Model and scaler saved to models/")

if __name__ == "__main__":
    train_model()