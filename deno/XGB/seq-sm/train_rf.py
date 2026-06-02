from utils.preprocessing import preprocess_and_split
from xgboost import XGBClassifier
import joblib

def train_xgb_pipeline():
    print("正在加载并预处理数据...")
    X_train, X_test, y_train, y_test = preprocess_and_split()

    print("正在训练 XGBoost 模型...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',  # 或 'mlogloss'，根据分类任务类型
        random_state=42
    )
    model.fit(X_train, y_train)

    print("正在保存模型...")
    joblib.dump(model, "models/xgb_model.pkl")

    print("训练完成!")

if __name__ == "__main__":
    train_xgb_pipeline()