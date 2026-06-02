import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import umap

# ✅ 步骤1：读取数据
df = pd.read_excel("GPT/CPP_encoded.xlsx", dtype={"CPP_Code": str})

# ✅ 步骤2：将 CPP_Code 转换为 20维向量
X = df["CPP_Code"].apply(lambda x: [int(i) for i in x.zfill(20)]).tolist()
X = np.array(X)

# ✅ 步骤3：使用 KMeans 聚类
kmeans = KMeans(n_clusters=2, random_state=42)
labels = kmeans.fit_predict(X)

# ✅ 步骤4：将聚类标签加入原始数据
df["Cluster"] = labels

# ✅ 步骤5：自定义颜色（注意顺序！）
custom_colors = [
    [95/255, 158/255, 160/255],   # 标签0: #5F9EA0 青色 (Non-CPP)
    [255/255, 165/255, 0/255]     # 标签1: #FFA500 橙色 (CPP)
]
custom_cmap = ListedColormap(custom_colors)

# ✅ 绘图函数（智能版）
def plot_embedding(X_embedded, labels, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(8, 6))
    plt.rcParams['font.family'] = 'Arial'
    scatter = plt.scatter(
        X_embedded[:, 0],
        X_embedded[:, 1],
        c=labels,
        cmap=custom_cmap,
        alpha=0.8,
        edgecolor='k',
        s=80
    )

    plt.xlabel(xlabel, fontweight='bold', fontsize=12)
    plt.ylabel(ylabel, fontweight='bold', fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontweight='bold', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # 图例：0对应Non-CPP，1对应CPP
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Non-CPP',
                  markerfacecolor=custom_colors[0], markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='CPP',
                  markerfacecolor=custom_colors[1], markersize=10)
    ]
    plt.legend(handles=legend_elements, loc = "upper right", framealpha=0.2,fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=1000, bbox_inches='tight')
    plt.show()

# ✅ 步骤6：PCA 可视化
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
plot_embedding(X_pca, labels, title="PCA (GPT)", xlabel="Component 1", ylabel="Component 2", save_path="ds_pca.png")

# ✅ 步骤7：t-SNE 可视化
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)
plot_embedding(X_tsne, labels, title="t-SNE (GPT)", xlabel="Component 1", ylabel="Component 2", save_path="ds_tsne.png")

# ✅ 步骤8：UMAP 可视化
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)
plot_embedding(X_umap, labels, title="UMAP (GTP)", xlabel="Component 1", ylabel="Component 2", save_path="ds_umap.png")

# ✅ 步骤9：保存带聚类标签的新Excel文件
df.to_excel("GPT/CPP_clustered.xlsx", index=False)
print("✅ 聚类和可视化全部完成！文件已保存")