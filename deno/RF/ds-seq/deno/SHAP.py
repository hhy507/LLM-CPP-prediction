import pandas as pd
import matplotlib.pyplot as plt


# 从生成的xlsx文件中读取数据
def load_peptide_predictions(excel_file="generated_peptides_with_prediction.xlsx"):
    df = pd.read_excel(excel_file)
    return df


# 绘制预测结果的条形图
def plot_peptides_probability(df):
    # 排序并取前三的肽
    top_peptides = df.sort_values(by="CPP Probability", ascending=False).head(3)

    # 创建条形图
    plt.figure(figsize=(10, 6))
    plt.barh(top_peptides['Peptide'], top_peptides['CPP Probability'], color='skyblue')
    plt.xlabel("CPP Probability")
    plt.title("Top 3 Peptides by CPP Probability")
    plt.gca().invert_yaxis()  # 反转Y轴，使得概率最高的肽排在顶部
    plt.tight_layout()
    plt.show()


# 输出前三的肽序列及其概率到txt文件
def output_top_peptides(df, filename="top_peptides.txt"):
    # 排序并取前三的肽
    top_peptides = df.sort_values(by="CPP Probability", ascending=False).head(3)

    with open(filename, 'w') as f:
        for idx, row in top_peptides.iterrows():
            f.write(f"Peptide: {row['Peptide']}, CPP Probability: {row['CPP Probability']:.4f}\n")

    print(f"✅ 已保存前三的肽序列到 {filename}")


# 主程序
def main():
    # 从文件加载数据
    df = load_peptide_predictions("generated_peptides_with_prediction.xlsx")

    # 绘制前3个预测结果的图表
    plot_peptides_probability(df)

    # 输出前三个肽序列到txt文件
    output_top_peptides(df, "top_peptides.txt")


if __name__ == "__main__":
    main()