import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from utils.preprocessing import load_and_preprocess_data

def train_model():
    print("🔍 Loading and extracting k-mer features...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(k=2)

    print("🚀 Training Random Forest model...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        oob_score=True
    )
    clf.fit(X_train, y_train)

    print("\n✅ Evaluation on test set:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/rf_kmer_model.pkl")
    print("✅ Model saved to models/rf_kmer_model.pkl")

if __name__ == "__main__":
    train_model()