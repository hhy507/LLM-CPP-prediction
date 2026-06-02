import os
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from utils.preprocessing import load_data


def train_model():
    X_train, X_test, y_train, y_test = load_data()

    print("Training XGBoost on sequence features only...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\nEvaluation on test set:")
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_model.pkl")
    print("Model saved to models/xgb_sequence_model.pkl")


if __name__ == "__main__":
    train_model()