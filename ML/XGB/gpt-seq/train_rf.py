import os
import joblib
import xgboost as xgb
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data

def train_model():
    # 加载数据
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 训练模型
    print("Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        scale_pos_weight=1,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # 评估模型
    print("\nModel Evaluation:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # 保存模型
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_model.pkl")
    print("Model saved to models/xgboost_model.pkl")

if __name__ == "__main__":
    train_model()