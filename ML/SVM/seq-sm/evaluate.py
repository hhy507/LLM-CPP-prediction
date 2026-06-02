# evaluate.py

import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.evaluation import ClassificationEvaluator
from utils.preprocessing import sequence_to_smiles, smiles_to_fingerprint

def load_and_predict(model_path="models/svm_model.pkl", test_path="data/test.xlsx", scaler_path="models/scaler.pkl"):
    df = pd.read_excel(test_path, converters={"Sequence": str})
    df["SMILES"] = df["Sequence"].apply(sequence_to_smiles)
    df["Fingerprint"] = df["SMILES"].apply(smiles_to_fingerprint)
    X = np.stack(df["Fingerprint"].values)
    y_true = df["Label"].values

    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    model = joblib.load(model_path)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    return y_true, y_pred, y_proba

if __name__ == "__main__":
    y_true, y_pred, y_proba = load_and_predict()

    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]
    )

    print("\n" + "=" * 50)
    print(" Evaluation Report ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for k, v in report.items():
        print(f"{k:<20}: {v:.4f}")

    print("\nGenerating visualizations...")
    evaluator.plot_all()