import random
import re
import numpy as np
import pandas as pd
import joblib
import shap
from tqdm import tqdm

# --- 氨基酸列表
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


# --- 随机生成肽
def random_peptide(length_range=(8, 30)):
    length = random.randint(*length_range)
    return ''.join(random.choices(AMINO_ACIDS, k=length))


# --- 计算多肽性质
def compute_peptide_properties(seq):
    return {
        'sequence': seq,
        'length': len(seq),
        'arg_count': seq.count('R'),
        'lys_count': seq.count('K'),
        'his_count': seq.count('H'),
        'pro_count': seq.count('P'),
        'gly_count': seq.count('G'),
        'aromatic_count': sum(seq.count(aa) for aa in 'FWY'),
        'cationic_count': sum(seq.count(aa) for aa in 'RKH'),
        'anionic_count': sum(seq.count(aa) for aa in 'DE'),
        'hydrophobicity': sum(seq.count(aa) for aa in 'AILMFWV') / max(1, len(seq)),
        'net_charge': (seq.count('R') + seq.count('K') + seq.count('H')) - (seq.count('D') + seq.count('E'))
    }


# --- 20条规则编码（返回规则通过的总数）
def encode_rules(p):
    hydrophobic_count = sum(p['sequence'].count(aa) for aa in 'AILMFWV')
    rules = [
        1 if p['arg_count'] >= 4 else 0,
        1 if p['lys_count'] > 0 else 0,
        1 if p['hydrophobicity'] < 0.4 else 0,
        1 if (0.3 < p['hydrophobicity'] < 0.6) else 0,
        1 if p['aromatic_count'] == 0 else 0,
        1 if 8 <= p['length'] <= 30 else 0,
        1 if (p['pro_count'] > 0 or p['gly_count'] > 0) else 0,
        1 if (0.8 <= p['cationic_count'] / max(1, hydrophobic_count) <= 1.2) else 0,
        1 if p['anionic_count'] == 0 else 0,
        1 if p['his_count'] > 0 else 0,
        1 if p['net_charge'] >= 4 else 0,
        1 if (0.2 <= p['hydrophobicity'] <= 0.4) else 0,
        1 if (p['arg_count'] + p['lys_count']) >= 4 else 0,
        1 if (len(p['sequence']) * 110) < 3000 else 0,
        1 if re.search(r'[RK][FILVWY]', p['sequence']) else 0,
        1 if re.search(r'[RK]{3,}', p['sequence']) else 0,
        1 if not re.search(r'[ED]{3,}', p['sequence']) else 0,
        1 if re.search(r'P.{1,2}P', p['sequence']) else 0,
        1 if p['sequence'].startswith('A') else 0,
        1 if re.search(r'C.{3,7}C', p['sequence']) else 0
    ]
    return sum(rules)


# --- 20维氨基酸频率
def encode_frequency(seq):
    return [seq.count(aa) / len(seq) if len(seq) > 0 else 0 for aa in AMINO_ACIDS]


# --- 生成符合规则的多肽
def generate_peptides_with_rules(n, length_range=(8, 30)):
    peptides = []
    pbar = tqdm(total=n, desc="Generating peptides")
    while len(peptides) < n:
        seq = random_peptide(length_range)
        p = compute_peptide_properties(seq)
        rules_count = encode_rules(p)
        if p['arg_count'] < 6:
            continue  # R不够，跳过

            # --- 再筛连续的R ---
        if not re.search(r'R{3,}', seq):  # 至少连续3个R
            continue
        if rules_count >= 15:  # 只要求通过15条以上
            peptides.append(seq)
            pbar.update(1)
    pbar.close()
    return peptides


# --- 主程序
def main():
    n_generate = 100
    model_path = "models/rf_model.pkl"  # 你的训练好的模型

    # 加载模型
    model = joblib.load(model_path)

    # 生成多肽
    peptides = generate_peptides_with_rules(n=n_generate)

    # 编码特征
    feature_list = []
    for seq in peptides:
        freq_feat = encode_frequency(seq)  # 20维氨基酸频率
        p = compute_peptide_properties(seq)
        rules_count = encode_rules(p)  # 1维规则总数
        feature = freq_feat + [rules_count]  # 合并为21维特征
        feature_list.append(feature)
    X = np.array(feature_list)

    # 预测
    preds = model.predict_proba(X)[:, 1]

    # shap解释
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # 保存结果
    df = pd.DataFrame({
        'Peptide': peptides,
        'CPP Probability': preds
    })

    # --- 处理 SHAP 值 ---
    shap_array = shap_values.values

    # 展平 SHAP 值，使其成为二维结构
    shap_array_reshaped = shap_array.reshape(shap_array.shape[0], -1)  # 将其转换为 (100, 21) 形状

    # 转换为 DataFrame
    shap_df = pd.DataFrame(shap_array_reshaped,
                           columns=[f'Feature_{i}_SHAP' for i in range(shap_array_reshaped.shape[1])])

    # 合并 SHAP 值到原始结果
    df = pd.concat([df, shap_df], axis=1)

    # 保存到 Excel
    df.to_excel("generated_peptides_with_prediction.xlsx", index=False)
    print("✅ 成功保存 generated_peptides_with_prediction.xlsx")


if __name__ == "__main__":
    main()