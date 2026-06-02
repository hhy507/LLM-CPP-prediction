import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import umap
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ================== 全局配置 ==================
AA_PROPERTIES = {
    'A': {'hydro': 1.8, 'volume': 88.6, 'charge': 0},
    'C': {'hydro': 2.5, 'volume': 108.5, 'charge': 0},
    'D': {'hydro': -3.5, 'volume': 111.1, 'charge': -1},
    'E': {'hydro': -3.5, 'volume': 138.4, 'charge': -1},
    'F': {'hydro': 2.8, 'volume': 189.9, 'charge': 0},
    'G': {'hydro': -0.4, 'volume': 60.1, 'charge': 0},
    'H': {'hydro': -3.2, 'volume': 153.2, 'charge': 1},
    'I': {'hydro': 4.5, 'volume': 166.7, 'charge': 0},
    'K': {'hydro': -3.9, 'volume': 168.6, 'charge': 1},
    'L': {'hydro': 3.8, 'volume': 166.7, 'charge': 0},
    'M': {'hydro': 1.9, 'volume': 162.9, 'charge': 0},
    'N': {'hydro': -3.5, 'volume': 114.1, 'charge': 0},
    'P': {'hydro': -1.6, 'volume': 112.7, 'charge': 0},
    'Q': {'hydro': -3.5, 'volume': 143.8, 'charge': 0},
    'R': {'hydro': -4.5, 'volume': 173.4, 'charge': 1},
    'S': {'hydro': -0.8, 'volume': 89.0, 'charge': 0},
    'T': {'hydro': -0.7, 'volume': 116.1, 'charge': 0},
    'V': {'hydro': 4.2, 'volume': 140.0, 'charge': 0},
    'W': {'hydro': -0.9, 'volume': 227.8, 'charge': 0},
    'Y': {'hydro': -1.3, 'volume': 193.6, 'charge': 0},
}

# 颜色统一：0-粉色（Non-CPP），1-蓝色（CPP）
CLASS_COLORS = {
    0: "#5F9EA0",  # 绿色
    1: "#FFA500"  # 橙色
}

CLASS_LABELS = {
    0: "Non-CPP",
    1: "CPP"
}

os.makedirs("figures", exist_ok=True)


# ================== 功能函数 ==================

def load_data(file_path):
    df = pd.read_excel(file_path)
    return df[["Sequence", "Label"]].dropna()


def compute_aa_frequencies(df):
    results = {}
    for label in sorted(df["Label"].unique()):
        counter = Counter()
        total = 0
        for seq in df[df["Label"] == label]["Sequence"]:
            for aa in seq:
                if aa in AA_PROPERTIES:
                    counter[aa] += 1
                    total += 1
        freqs = {aa: counter.get(aa, 0) / total for aa in AA_PROPERTIES}
        results[label] = freqs
    return results


def plot_aa_frequencies(freqs_by_class):
    aa = list(AA_PROPERTIES.keys())
    df_plot = pd.DataFrame({
        "Amino Acid": aa,
        "Non-CPP": [freqs_by_class[0][a] for a in aa],
        "CPP": [freqs_by_class[1][a] for a in aa]
    }).melt(id_vars="Amino Acid", var_name="Class", value_name="Frequency")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_plot,
        x="Amino Acid",
        y="Frequency",
        hue="Class",
        palette=[CLASS_COLORS[0], CLASS_COLORS[1]]
    )
    plt.xlabel("Amino Acid", fontsize=16, fontweight="bold")
    plt.ylabel("Frequency", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, loc = "upper right")
    plt.tight_layout()
    plt.savefig("figures/aa_frequency_by_label.png", dpi=300)
    plt.close()


def compute_physicochemical(df):
    rows = []
    for _, row in df.iterrows():
        seq = row["Sequence"]
        label = row["Label"]
        props = {key: [AA_PROPERTIES[aa][key] for aa in seq if aa in AA_PROPERTIES] for key in
                 ['hydro', 'volume', 'charge']}
        avg_props = {key: np.mean(values) if values else 0 for key, values in props.items()}
        avg_props["Label"] = label
        rows.append(avg_props)
    return pd.DataFrame(rows)


def plot_physicochemical(df_props):
    for prop in ['hydro', 'volume', 'charge']:
        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=df_props,
            x="Label",
            y=prop,
            palette=[CLASS_COLORS[0], CLASS_COLORS[1]],
            cut=0,
            linewidth=1.2
        )
        plt.xlabel("Label", fontsize=16, fontweight="bold")
        plt.ylabel(prop.capitalize(), fontsize=16, fontweight="bold")
        plt.xticks(
            ticks=[0, 1],
            labels=[CLASS_LABELS[0], CLASS_LABELS[1]],
            fontsize=12,
            fontweight="bold"
        )
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(f"figures/{prop}_violin_by_label.png", dpi=300)
        plt.close()


def plot_onehot_matrix(df, n_samples=10):
    aa_list = list(AA_PROPERTIES.keys())
    encoder = OneHotEncoder(categories=[aa_list], handle_unknown='ignore', sparse_output=False)

    for label in sorted(df["Label"].unique()):
        sequences = df[df["Label"] == label]["Sequence"].tolist()[:n_samples]
        max_len = max(len(seq) for seq in sequences)
        matrices = []
        for seq in sequences:
            encoded = encoder.fit_transform(np.array(list(seq)).reshape(-1, 1))
            padded = np.pad(encoded, ((0, max_len - len(seq)), (0, 0)), mode='constant')
            matrices.append(padded)
        matrix = np.vstack(matrices)
        plt.figure(figsize=(12, 6))
        sns.heatmap(matrix, cmap="viridis")
        plt.title(f"One-Hot Matrix - {CLASS_LABELS[label]}")
        plt.xlabel("Amino Acids")
        plt.ylabel("Position")
        plt.tight_layout()
        plt.savefig(f"figures/onehot_label_{label}.png", dpi=300)
        plt.close()


def compute_kmer_frequencies(df, k=2):
    results = {}
    for label in sorted(df["Label"].unique()):
        counter = Counter()
        for seq in df[df["Label"] == label]["Sequence"]:
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i + k]
                counter[kmer] += 1
        results[label] = dict(counter.most_common(20))
    return results


def plot_kmer_frequencies(kmer_freqs):
    kmers = sorted(set(kmer_freqs[0]) | set(kmer_freqs[1]))
    df_plot = pd.DataFrame({
        "K-mer": kmers,
        "Non-CPP": [kmer_freqs[0].get(k, 0) for k in kmers],
        "CPP": [kmer_freqs[1].get(k, 0) for k in kmers]
    }).melt(id_vars="K-mer", var_name="Class", value_name="Count")

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=df_plot,
        x="K-mer",
        y="Count",
        hue="Class",
        palette=[CLASS_COLORS[0], CLASS_COLORS[1]]
    )
    plt.xlabel("K-mer", fontsize=16, fontweight="bold")
    plt.ylabel("Count", fontsize=16, fontweight="bold")
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12, title_fontsize=12,loc = "upper right")
    plt.tight_layout()
    plt.savefig("figures/kmer2_by_label.png", dpi=300)
    plt.close()


def plot_sequence_length_distribution(df):
    df["Length"] = df["Sequence"].apply(len)
    custom_colors = ["#5F9EA0", "#FFA500"]
    plt.figure(figsize=(8, 6))
    for label, color in zip([0, 1], custom_colors):
        lengths = df[df["Label"] == label]["Length"]
        density = gaussian_kde(lengths)
        xs = np.linspace(lengths.min() - 5, lengths.max() + 5, 200)
        plt.plot(xs, density(xs), label="Non-CPP" if label == 0 else "CPP", color=color, linewidth=2)

    plt.xlabel("Sequence Length", fontweight="bold", fontsize=18)
    plt.ylabel("Density", fontweight="bold", fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("figures/sequence_length_distribution.png", dpi=300)
    plt.close()

def prepare_features(df, method="physicochemical"):
    features = []
    for seq in df["Sequence"]:
        if method == "physicochemical":
            props = {key: [AA_PROPERTIES[aa][key] for aa in seq if aa in AA_PROPERTIES] for key in ['hydro', 'volume', 'charge']}
            features.append([
                np.mean(props["hydro"]) if props["hydro"] else 0,
                np.mean(props["volume"]) if props["volume"] else 0,
                np.mean(props["charge"]) if props["charge"] else 0,
            ])
        elif method == "aa_freq":
            counter = Counter(seq)
            features.append([counter.get(aa, 0)/len(seq) for aa in AA_PROPERTIES.keys()])
    return np.array(features)

def plot_embedding(df, method="tsne", feature_type="physicochemical"):
    X = prepare_features(df, method=feature_type)
    y = df["Label"].values

    if method == "tsne":
        embedder = TSNE(n_components=2, random_state=42, perplexity=30)
    elif method == "umap":
        embedder = umap.UMAP(random_state=42)

    X_embedded = embedder.fit_transform(X)
    custom_colors = ["#5F9EA0", "#FFA500"]
    plt.figure(figsize=(8,6))
    for label, color in zip([0, 1], custom_colors):
        idx = y == label
        plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],
                    label="Non-CPP" if label==0 else "CPP",
                    color=color, alpha=0.7, edgecolors='k', s=60)

    plt.xlabel("Component 1", fontweight="bold", fontsize=12)
    plt.ylabel("Component 2", fontweight="bold", fontsize=12)
    plt.title(f"{method.upper()} ({feature_type})", fontweight="bold", fontsize=14)
    plt.legend(fontsize=14, framealpha=0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"figures/{method}_{feature_type}_embedding.png", dpi=300)
    plt.close()

def plot_pca(df, feature_type="physicochemical"):
    X = prepare_features(df, method=feature_type)
    y = df["Label"].values

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    custom_colors = ["#5F9EA0", "#FFA500"]

    plt.figure(figsize=(8,6))
    for label, color in zip([0, 1], custom_colors):
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1],
                    label="Non-CPP" if label==0 else "CPP",
                    color=color, alpha=0.7, edgecolors='k', s=60)

    plt.xlabel("Component 1", fontweight="bold", fontsize=12)
    plt.ylabel("Component 2", fontweight="bold", fontsize=12)
    plt.title(f"PCA ({feature_type})", fontweight="bold", fontsize=14)
    plt.legend(fontsize=14,framealpha=0.2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(f"figures/pca_{feature_type}_embedding.png", dpi=300)
    plt.close()




# ================== 主程序 ==================

if __name__ == "__main__":
    file_path = "data/train.xlsx"  # <-- 换成你的路径

    df = load_data(file_path)

    print(">> 生成氨基酸频率柱状图...")
    freqs_by_class = compute_aa_frequencies(df)
    plot_aa_frequencies(freqs_by_class)

    print(">> 生成理化性质小提琴图...")
    props_df = compute_physicochemical(df)
    plot_physicochemical(props_df)

    print(">> 生成One-Hot编码热图...")
    plot_onehot_matrix(df)

    print(">> 生成K-mer频率柱状图...")
    kmer_freqs = compute_kmer_frequencies(df)
    plot_kmer_frequencies(kmer_freqs)

    print(">> Plotting Sequence Length Distribution...")
    plot_sequence_length_distribution(df)

    print(">> Plotting t-SNE and UMAP Embeddings...")
    plot_embedding(df, method="tsne", feature_type="physicochemical")
    plot_embedding(df, method="umap", feature_type="physicochemical")
    plot_embedding(df, method="tsne", feature_type="aa_freq")
    plot_embedding(df, method="umap", feature_type="aa_freq")

    print(">> Plotting PCA Embeddings...")
    plot_pca(df, feature_type="physicochemical")
    plot_pca(df, feature_type="aa_freq")

    print("✅ 全部图形已保存在 'figures/' 文件夹中。")