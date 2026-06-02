import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# --- 配置 ---
excel_file = 'data/RF.xlsx'  # 换成你的文件路径
index_col = 0  # 索引列
color_hex = 0xABBBDD  # 主色调（浅蓝）



def create_cmap():
    colors = [
        (0.0, (214/255, 228/255, 240/255)),  # #D6E4F0 柔浅蓝
        (0.5, (123/255, 175/255, 212/255)),  # #7BAFD4 中柔蓝
        (1.0, (32/255, 93/255, 140/255))     # #205D8C 深科技蓝
    ]
    return LinearSegmentedColormap.from_list("soft_science_blue", colors, N=256)

# --- 读取数据 ---
df = pd.read_excel(excel_file, index_col=index_col)

# --- 绘制热图 ---
fig, ax = plt.subplots(figsize=(2 + df.shape[1]*1.5, 1 + df.shape[0]*0.8))

cmap = create_cmap()
im = ax.imshow(df.values, cmap=cmap)

# --- 写数值 ---
for i in range(df.shape[0]):
    for j in range(df.shape[1]):
        val = df.values[i, j]
        font_color = 'black' if val < (df.max().max() * 0.5) else 'white'
        ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                fontsize=14, color=font_color, fontweight='bold')

# --- 美化边框与轴 ---
ax.set_xticks(np.arange(df.shape[1]))
ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=14, fontweight='bold')
ax.set_yticks(np.arange(df.shape[0]))
ax.set_yticklabels(df.index, fontsize=14, fontweight='bold')

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xticks(np.arange(df.shape[1]+1)-0.5, minor=True)
ax.set_yticks(np.arange(df.shape[0]+1)-0.5, minor=True)
ax.grid(which="minor", color="black", linewidth=0.5)
ax.tick_params(which="minor", bottom=False, left=False)

plt.savefig('output/RF_heatmap.png', dpi=600, bbox_inches='tight')