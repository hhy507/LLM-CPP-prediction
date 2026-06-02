import shap
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'

# 载入模型
def load_model(model_path="models/lr_combined_model.pkl"):
    return joblib.load(model_path)

# 载入肽预测数据
def load_peptide_predictions(excel_file="generated_peptides_with_prediction.xlsx"):
    df = pd.read_excel(excel_file)
    return df

# 编码肽序列为定长特征向量
def encode_sequence(seq, max_length=40):
    """确保每个序列编码为定长的特征向量，长度不足时补零，过长时截断"""
    seq = seq.upper()
    encoded = [AMINO_ACIDS.index(aa) for aa in seq if aa in AMINO_ACIDS]
    # 填充到40个元素，如果不足就补零
    encoded = encoded[:max_length] + [0] * (max_length - len(encoded))
    return encoded

# 计算SHAP值
def calculate_shap_values(model, X):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    return shap_values

# 绘制SHAP特征重要性图
def plot_shap_feature_importance(shap_values):
    # SHAP值特征重要性图
    shap.summary_plot(shap_values, show=True)

# 绘制SHAP单个样本的贡献图
def plot_shap_single_sample(shap_values, base_value, sample_idx=0):
    # SHAP图 - 单个样本的贡献图
    shap.initjs()  # 启动shap的JavaScript可视化功能
    shap.plots.force(base_value, shap_values[sample_idx], show=True)

# 主程序
def main():
    model_path = "models/lr_combined_model.pkl"
    excel_file = "generated_peptides_with_prediction.xlsx"

    # 载入数据和模型
    df = load_peptide_predictions(excel_file)
    model = load_model(model_path)

    # 编码肽序列为40维特征
    peptides_encoded = np.array([encode_sequence(seq) for seq in df['Peptide']])

    # 计算SHAP值
    shap_values = calculate_shap_values(model, peptides_encoded)

    # 获取模型的预期值（用于force_plot）
    base_value = shap_values.base_values

    # 绘制特征重要性图
    plot_shap_feature_importance(shap_values)

    # 绘制一个样本的SHAP值贡献图
    plot_shap_single_sample(shap_values, base_value, sample_idx=0)  # 可以修改索引选择不同的样本

if __name__ == "__main__":
    main()