import pandas as pd


def bio_txt_to_xlsx(input_file, output_file):
    """
    将生物信息学格式的txt转换为xlsx
    格式示例：>id|label\nSEQUENCE
    """
    records = []
    with open(input_file, 'r') as f:
        current_id, current_label, current_seq = None, None, None
        for line in f:
            if line.startswith('>'):
                # 解析头部：>id|label
                header = line[1:].strip().split('|')
                current_id = header[0]
                current_label = header[1] if len(header) > 1 else None
            else:
                # 捕获序列
                current_seq = line.strip()
                records.append({
                    'ID': current_id,
                    'Label': current_label,
                    'Sequence': current_seq
                })

    # 转换为DataFrame并保存
    df = pd.DataFrame(records)
    df.to_excel(output_file, index=False)


# 使用示例
bio_txt_to_xlsx('./CPP924.txt', './CPP924.xlsx')
