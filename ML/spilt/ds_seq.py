import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split

# 读取 Excel 文件
df = pd.read_excel('./ds/CPP_encoded.xlsx')

# 处理 CPP_Code 列（字符串转为列表）
def convert_properties(value):
    if isinstance(value, str):
        return eval(value)
    return value

df['CPP_Code'] = df['CPP_Code'].apply(convert_properties)

# 最大氨基酸序列长度
MAX_SEQ_LENGTH = 50

# 氨基酸转数字 + 填充
def sequence_to_numeric(sequence):
    mapping = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    numeric_seq = [mapping.get(aa, -1) for aa in sequence]
    if len(numeric_seq) < MAX_SEQ_LENGTH:
        numeric_seq += [-1] * (MAX_SEQ_LENGTH - len(numeric_seq))
    return numeric_seq[:MAX_SEQ_LENGTH]

df['amino_acid_numeric'] = df['Sequence'].apply(sequence_to_numeric)

# 合并特征向量
X = np.array([df['amino_acid_numeric'][i] + df['CPP_Code'][i] for i in range(len(df))])
y = df['Label'].values
indices = np.arange(len(df))  # 保留原始索引

# 划分数据
X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
    X, y, indices, test_size=0.2, random_state=42
)

# ===== 验证集 =====
val_data = pd.DataFrame(X_val)
val_data['Sequence'] = df.loc[idx_val, 'Sequence'].values
val_data['CPP_Code'] = df.loc[idx_val, 'CPP_Code'].values
val_data['Label'] = y_val
val_data['CPP_Code'] = val_data['CPP_Code'].apply(json.dumps)  # 转为JSON字符串
val_data.to_excel('./ml/ds_seq/ds_seq_val.xlsx', index=False)

# ===== 训练集 =====
train_data = pd.DataFrame(X_train)
train_data['Sequence'] = df.loc[idx_train, 'Sequence'].values
train_data['CPP_Code'] = df.loc[idx_train, 'CPP_Code'].values
train_data['Label'] = y_train
train_data['CPP_Code'] = train_data['CPP_Code'].apply(json.dumps)  # 同样转为JSON字符串
train_data.to_excel('./ml/ds_seq/ds_seq_train.xlsx', index=False)

print("✅ 训练集和验证集已成功保存为 Excel 文件（CPP_Code 类型保留）")