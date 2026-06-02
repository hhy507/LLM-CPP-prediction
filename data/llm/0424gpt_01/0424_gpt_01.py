import pandas as pd
import re

# 简化理化属性估算函数
def simple_pI_estimate(seq):
    pos = seq.count('R') + seq.count('K') + seq.count('H')
    neg = seq.count('D') + seq.count('E')
    return 10 if pos > neg else 6

def simple_net_charge(seq):
    pos = seq.count('R') + seq.count('K') + 0.1 * seq.count('H')
    neg = seq.count('D') + seq.count('E')
    return pos - neg

def compute_cpp_code(seq):
    seq = seq.upper()
    length = len(seq)
    arg_count = seq.count('R')
    lys_count = seq.count('K')
    asp_glu_count = seq.count('D') + seq.count('E')
    trp_count = seq.count('W')
    his_count = seq.count('H')
    hydrophobic_residues = sum(seq.count(aa) for aa in "FILVWY")
    net_charge = simple_net_charge(seq)
    pI = simple_pI_estimate(seq)
    mol_weight = length * 110  # 平均每个氨基酸约110 Da
    hydrophobicity = (seq.count('A') + hydrophobic_residues - asp_glu_count) / length

    rules = [
        1 if arg_count >= 6 else 0,  # 规则1
        1 if trp_count > 0 else 0,  # 规则2
        1 if 8 <= arg_count <= 12 else 0,  # 规则3
        1 if his_count > 0 else 0,  # 规则4
        1 if arg_count > lys_count else 0,  # 规则5
        0 if asp_glu_count > 0 else 1,  # 规则6
        1 if 1 <= hydrophobic_residues <= 10 else 0,  # 规则7
        1 if re.search(r'[RK][FILVWY]', seq) else 0,  # 规则8
        1 if re.search(r'R.{1,2}R', seq) else 0,  # 规则9
        1,  # 规则10 假设N端乙酰化C端酰胺化
        1 if net_charge >= 4 else 0,  # 规则11
        1 if 'R' in seq and 'F' in seq else 0,  # 规则12：简化疏水矩代理
        1 if mol_weight < 5000 else 0,  # 规则13
        1 if pI > 9 else 0,  # 规则14
        1 if -1 < hydrophobicity < 1 else 0,  # 规则15
        1,  # 规则16 假设稳定
        1 if re.search(r'[AILMKRHFYVW]{4,}', seq) else 0,  # 规则17
        1 if seq[0] == seq[-1] else 0,  # 规则18 简化环化判断
        1 if length < 35 else 0,  # 规则19
        1 if 'GG' not in seq else 0  # 规则20 聚集回避
    ]
    return ''.join(map(str, rules))

# 读取Excel文件
df = pd.read_excel("./CPP924.xlsx")

# 应用规则并生成编码
df["CPP_Code"] = df["Sequence"].apply(compute_cpp_code)

# 保存到新的Excel文件
df.to_excel("./CPP_encoded.xlsx", index=False)

print("✅ 编码完成，结果已保存为 CPP_encoded.xlsx")