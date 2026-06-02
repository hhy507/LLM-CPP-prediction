import pandas as pd

# 1. 读取Excel文件
file_path = input("./ds/CPP_encoded.xlsx").strip()
df = pd.read_excel(file_path)

# 2. 检查CPP_Code列是否存在
if "CPP_Code" not in df.columns:
    print("错误：Excel文件中没有 'CPP_Code' 列！")
else:
    # 3. 处理每一行的二进制编码
    def calculate_binary_sum(binary_str):
        if pd.isna(binary_str):  # 处理空值
            return None
        binary_str = str(binary_str).strip()  # 转为字符串并去除空格
        # 补全前导零到20位
        binary_padded = binary_str.zfill(20)[:20]  # 确保20位
        # 检查是否全是0和1
        if not set(binary_padded).issubset({'0', '1'}):
            return None  # 非法输入
        return sum(int(bit) for bit in binary_padded)

    # 4. 计算总和并存储到新列
    df['CPP_Sum'] = df['CPP_Code'].apply(calculate_binary_sum)

    # 5. 保存结果到新Excel文件
    output_path = file_path.replace(".xlsx", "_with_sum.xlsx")
    df.to_excel(output_path, index=False)
    print(f"处理完成！结果已保存到: {output_path}")