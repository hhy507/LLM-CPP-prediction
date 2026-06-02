# train_lr.py

from utils.preprocessing import preprocess_and_split, train_and_save_lr
from sklearn.metrics import classification_report
import joblib

def train_logistic_regression():
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_and_split()

    print("Training Logistic Regression...")
    train_and_save_lr(X_train, y_train)

    print("Evaluating on test set...")
    model = joblib.load("models/logistic_regression_model.pkl")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Non-CPP", "CPP"]))

if __name__ == "__main__":
    train_logistic_regression()