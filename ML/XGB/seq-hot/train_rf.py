# train_onehot.py

import joblib
import xgboost as xgb
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_onehot

def train_onehot_xgb(model_path="models/xgb_onehot_model.pkl", max_len=50):
    X_train, X_test, y_train, y_test = load_and_preprocess_onehot(max_len=max_len)

    model = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"])
    print("XGBoost 模型训练完成，性能如下：")
    print(report)

if __name__ == "__main__":
    train_onehot_xgb()