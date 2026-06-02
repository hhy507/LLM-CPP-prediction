import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import colors

# --- 配置 ---
excel_file = 'data/XGB.xlsx'  # 换成你的文件路径
index_col = 0
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# --- 读取数据 ---
df = pd.read_excel(excel_file, index_col=index_col)

# --- 标准化数值为比例 ---
max_val = df.values.max()
proportions = df / max_val  # 将每个数值转换为0~1之间

# 创建蓝色渐变 cmap
cmap = plt.cm.Blues
norm = colors.Normalize(vmin=df.values.min(), vmax=df.values.max())

# --- 绘制小饼图矩阵 ---
fig, ax = plt.subplots(figsize=(1.5 * df.shape[1], 1.5 * df.shape[0]))

for i, row_label in enumerate(df.index):
    for j, col_label in enumerate(df.columns):
        val = proportions.iloc[i, j]
        # 蓝色根据原始值深浅，灰色为剩余
        blue_color = cmap(norm(df.iloc[i, j]))
        sizes = [val, 1 - val]
        colors_list = [blue_color, '#E0E0E0']

        # 创建小坐标轴
        ax_inset = plt.axes([j / df.shape[1], 1 - (i + 1) / df.shape[0], 1 / df.shape[1], 1 / df.shape[0]])
        ax_inset.pie(sizes, colors=colors_list, startangle=90, counterclock=False,
                     wedgeprops={'linewidth': 0})
        ax_inset.set_aspect('equal')
        ax_inset.axis('off')

        # 在饼图中心显示原始数值
        ax_inset.text(0, 0, f"{df.iloc[i, j]:.2f}", ha='center', va='center', fontsize=12, color='white')

# 去掉主坐标轴
ax.axis('off')
plt.tight_layout()
plt.savefig(f'{output_dir}/XGB_cell_piechart_blue.png', dpi=600)
plt.show()