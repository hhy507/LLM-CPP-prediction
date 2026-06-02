import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 读取原始数据
df = pd.read_excel("./gpt/CPP_encoded.xlsx", dtype={"CPP_Code": str})

# 提取特征和标签
X = df["CPP_Code"].apply(lambda x: [int(i) for i in x.zfill(20)]).tolist()  # 仅用于拆分
X = np.array(X)
y = df["Label"]  # 确保这一列是你的01标签
original_codes = df["CPP_Code"]  # 保留原始CPP编码

# 拆分数据（保持原始CPP编码）
_, _, _, _, idx_train, idx_test = train_test_split(
    X, y, original_codes.index,  # 传入原始索引
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 构建结果DataFrame（只保留label和CPP_Code）
train_df = pd.DataFrame({
    "label": y.loc[idx_train].values,
    "CPP_Code": original_codes.loc[idx_train].values
})

test_df = pd.DataFrame({
    "label": y.loc[idx_test].values,
    "CPP_Code": original_codes.loc[idx_test].values
})

# 保存到Excel文件（保持原始格式）
train_df.to_excel("./ml/gpt/gpt_train.xlsx", index=False)
test_df.to_excel("./ml/gpt/gpt_test.xlsx", index=False)

print("✅ 数据拆分完成！")
print(f"训练集样本数: {len(train_df)} | 测试集样本数: {len(test_df)}")
print(f"训练集标签分布:\n{train_df['label'].value_counts()}")
print(f"测试集标签分布:\n{test_df['label'].value_counts()}")