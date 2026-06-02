import pandas as pd


def filter_non_empty_rows(input_file, output_file):
    """
    从Excel中筛选出所有不包含空值的行

    参数:
        input_file (str): 输入Excel文件路径
        output_file (str): 输出Excel文件路径
    """
    # 读取Excel文件，将各种空值标记转换为NaN
    df = pd.read_excel(input_file, na_values=['', ' ', 'NA', 'NaN', '.', 'NULL'])

    # 筛选出所有不包含NaN的行
    non_empty_df = df.dropna(how='any')

    # 打印统计信息
    print(f"原始数据行数: {len(df)}")
    print(f"无空值行数: {len(non_empty_df)}")
    print(f"已删除行数: {len(df) - len(non_empty_df)}")

    # 保存到新文件
    non_empty_df.to_excel(output_file, index=False)
    print(f"结果已保存到: {output_file}")


# 使用示例
if __name__ == "__main__":
    input_excel = "./PCA1.xlsx"  # 替换为你的输入文件
    output_excel = "./PCA2.xlsx"  # 输出文件名

    filter_non_empty_rows(input_excel, output_excel)