import pandas as pd
from sklearn.model_selection import train_test_split

# 读取你的原始文件（假设列为 'Sequence' 和 'Label'）
df = pd.read_excel("./ml/seq/seq.xlsx")

# 确保列名正确
assert "Sequence" in df.columns and "Label" in df.columns, "请确认列名包含 'Sequence' 和 'Label'"

# 拆分数据集（按标签 stratify）
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["Label"]
)

# 保存为 Excel 文件
train_df.to_excel("./ml/seq/seq_train.xlsx", index=False)
test_df.to_excel("./ml/seq/seq_test.xlsx", index=False)

print("✅ 按序列+标签拆分完成")
print("训练集：train_sequences.xlsx")
print("测试集：test_sequences.xlsx")