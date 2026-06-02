import pandas as pd
import numpy as np
import joblib
from utils.preprocessing import sequence_to_smiles, smiles_to_fingerprint
from sklearn.metrics import classification_report
from utils.evaluation import ClassificationEvaluator

def load_and_predict(model_path="models/xgb_model.pkl", test_path="data/test.xlsx", scaler_path="models/xgb_scaler.pkl"):
    """ 加载模型，进行预测并返回结果 """
    df = pd.read_excel(test_path, converters={"Sequence": str})
    df["SMILES"] = df["Sequence"].apply(sequence_to_smiles)
    df["Fingerprint"] = df["SMILES"].apply(smiles_to_fingerprint)
    X = np.stack(df["Fingerprint"].values)
    y_true = df["Label"].values

    # 使用保存的 scaler 对测试数据进行标准化
    scaler = joblib.load(scaler_path)
    X_scaled = scaler.transform(X)

    # 加载 XGBoost 模型并预测
    model = joblib.load(model_path)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    return y_true, y_pred, y_proba

def evaluate_model():
    """ 评估模型并输出结果 """
    y_true, y_pred, y_proba = load_and_predict()

    evaluator = ClassificationEvaluator(
        y_true=y_true,
        y_pred=y_pred,
        y_proba=y_proba,
        class_names=["Non-CPP", "CPP"]
    )

    print("\n" + "=" * 50)
    print(" XGBoost 模型评估 ".center(50, '='))
    print("=" * 50)
    report = evaluator.generate_report()
    for k, v in report.items():
        print(f"{k:<20}: {v:.4f}")

    print("\n生成可视化结果...")
    evaluator.plot_all()

if __name__ == "__main__":
    evaluate_model()