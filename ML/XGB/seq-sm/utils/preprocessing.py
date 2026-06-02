import os
import numpy as np
import pandas as pd
import joblib  # 添加 joblib 导入
from rdkit import Chem
from rdkit.Chem import AllChem  # 使用 AllChem 来生成指纹
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  # 导入 train_test_split

# 氨基酸 SMILES 对应表
AMINO_ACID_SMILES = {
    'A': 'CC(C(=O)O)N', 'C': 'C(C(=O)O)N', 'D': 'CC(C(=O)O)N',
    'E': 'CCC(C(=O)O)N', 'F': 'CC1=CC=CC=C1C(=O)O', 'G': 'C(C(=O)O)N',
    'H': 'CC1=CNC=N1', 'I': 'CC(C)CC(=O)O', 'K': 'CCCC(C(=O)O)N',
    'L': 'CC(C)C(=O)O', 'M': 'CCSC(=O)O', 'N': 'CC(C(=O)O)N',
    'P': 'CC1CC1C(=O)O', 'Q': 'CCC(C(=O)O)N', 'R': 'CCCNC(=N)N',
    'S': 'COC(=O)O', 'T': 'CC(O)C(=O)O', 'V': 'CC(C)C(=O)O',
    'W': 'CC1=CNC2=CC=CC=C12', 'Y': 'CC1=CC=C(O)C=C1'
}


def sequence_to_smiles(seq):
    """ 将氨基酸序列转换为 SMILES 表示 """
    return '.'.join([AMINO_ACID_SMILES.get(aa, '') for aa in seq if aa in AMINO_ACID_SMILES])


def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    """ 将 SMILES 转换为指纹（使用 AllChem） """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits)

    # 使用 AllChem 获取摩根指纹
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)

    # 将指纹转换为 numpy 数组
    return np.array(fp)


def preprocess_and_split(data_path="data/train.xlsx", test_size=0.2, random_state=42):
    """ 加载数据、预处理并分割训练集和测试集 """
    df = pd.read_excel(data_path, converters={"Sequence": str})
    df["SMILES"] = df["Sequence"].apply(sequence_to_smiles)
    df["Fingerprint"] = df["SMILES"].apply(smiles_to_fingerprint)
    X = np.stack(df["Fingerprint"].values)
    y = df["Label"].values

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/xgb_scaler.pkl")  # 使用 joblib 保存 scaler

    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)  # 这里添加了导入的 train_test_split