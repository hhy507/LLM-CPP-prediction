# train.py

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data

def train_model():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        class_weight='balanced',
        oob_score=True
    )
    model.fit(X_train, y_train)

    print("\nEvaluation on Test Set:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/rf_model.pkl")
    print("Model saved to models/rf_model.pkl")

if __name__ == "__main__":
    train_model()