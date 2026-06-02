import pandas as pd
import joblib
from utils.preprocessing import extract_kmer_features
from utils.evaluation import ClassificationEvaluator

def load_test_data_and_model():
    test_df = pd.read_excel("data/test.xlsx", converters={"Sequence": str})
    sequences = test_df["Sequence"].astype(str).tolist()
    y_true = test_df["Label"].values

    X_test = extract_kmer_features(sequences, k=2)

    model = joblib.load("models/rf_kmer_model.pkl")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    return y_true, y_pred, y_proba

if __name__ == "__main__":
    y_true, y_pred, y_proba = load_test_data_and_model()

    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]
    )

    print("\n" + "=" * 50)
    print(" Quantitative Metrics ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for metric, value in report.items():
        print(f"{metric:<25}: {value:.4f}")

    print("\nGenerating visualizations...")
    evaluator.plot_all()