import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
import warnings
import os
from datetime import datetime

# 设置样式和忽略警告
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')


class PeptideAnalyzer:
    def __init__(self, file_path, output_dir='results'):
        self.file_path = file_path
        self.output_dir = output_dir
        self.create_output_dir()
        self.ids = None
        self.features = None
        self.X_scaled = None
        self.X_pca = None
        self.X_umap = None
        self.kmeans_labels = None
        self.dbscan_labels = None
        self.hierarchical_labels = None
        self.current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    def create_output_dir(self):
        """创建输出目录"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def load_data(self):
        """从Excel文件加载数据并处理缺失值"""
        try:
            df = pd.read_excel(self.file_path, na_values=['', 'NA', 'NaN', '.', 'NULL'])
            df = df.apply(pd.to_numeric, errors='coerce').fillna(df.mean())

            print(f"数据加载成功，样本数: {len(df)}, 特征数: {len(df.columns) - 1}")
            print("前5行数据:\n", df.head())

            self.ids = df.iloc[:, 0] if len(df.columns) > 1 else None
            self.features = df.iloc[:, 1:] if len(df.columns) > 1 else df
            return True
        except Exception as e:
            print(f"加载数据失败: {str(e)}")
            return False

    def preprocess_data(self):
        """数据标准化"""
        try:
            scaler = StandardScaler()
            self.X_scaled = scaler.fit_transform(self.features)
            print("数据标准化完成")
            return True
        except Exception as e:
            print(f"数据预处理失败: {str(e)}")
            return False

    def apply_dimensionality_reduction(self):
        """PCA和UMAP降维"""
        try:
            # PCA
            pca = PCA(n_components=0.95)
            self.X_pca = pca.fit_transform(self.X_scaled)
            print(f"PCA降维完成，保留主成分数: {pca.n_components_}")
            print("各主成分解释方差比:", pca.explained_variance_ratio_)

            # UMAP
            umap = UMAP(n_components=2, n_neighbors=min(20, len(self.X_scaled) - 1),
                        min_dist=0.1, random_state=42)
            self.X_umap = umap.fit_transform(self.X_scaled)
            print("UMAP降维完成")
            return True
        except Exception as e:
            print(f"降维失败: {str(e)}")
            return False

    def apply_clustering(self):
        """K-Means、DBSCAN和层次聚类"""
        try:
            # K-Means
            optimal_k = self.find_optimal_k()
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            self.kmeans_labels = kmeans.fit_predict(self.X_pca)
            print(f"K-Means聚类完成，使用K={optimal_k}")

            # DBSCAN
            dbscan = DBSCAN(eps=0.5, min_samples=5)
            self.dbscan_labels = dbscan.fit_predict(self.X_umap)
            n_clusters = len(set(self.dbscan_labels)) - (1 if -1 in self.dbscan_labels else 0)
            print(f"DBSCAN聚类完成，发现簇数: {n_clusters}")

            # 层次聚类
            hierarchical = AgglomerativeClustering(n_clusters=optimal_k,
                                                   metric='euclidean',
                                                   linkage='ward')
            self.hierarchical_labels = hierarchical.fit_predict(self.X_pca)
            print("层次聚类完成")
            return True
        except Exception as e:
            print(f"聚类失败: {str(e)}")
            return False

    def find_optimal_k(self, max_k=10):
        """通过轮廓系数和肘部法则确定最佳K值"""
        silhouette_scores = []
        distortions = []
        K_range = range(2, max_k + 1)

        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(self.X_pca)
            silhouette_scores.append(silhouette_score(self.X_pca, labels))
            distortions.append(kmeans.inertia_)

        optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
        diff = np.diff(distortions)
        optimal_k_elbow = K_range[np.argmin(diff[1:] / diff[:-1]) + 2]

        optimal_k = (optimal_k_silhouette + optimal_k_elbow) // 2
        print(f"建议K值: 轮廓系数法={optimal_k_silhouette}, 肘部法则={optimal_k_elbow}, 最终选择={optimal_k}")
        return optimal_k

    def evaluate_clusters(self):
        """评估聚类质量"""
        metrics = {}
        for name, labels in [('K-Means', self.kmeans_labels),
                             ('Hierarchical', self.hierarchical_labels)]:
            metrics[name] = {
                'Silhouette': silhouette_score(self.X_pca, labels),
                'Calinski-Harabasz': calinski_harabasz_score(self.X_pca, labels),
                'Davies-Bouldin': davies_bouldin_score(self.X_pca, labels)
            }

        print("\n聚类评估结果:")
        for method, scores in metrics.items():
            print(f"\n{method}:")
            for k, v in scores.items():
                print(f"  {k}: {v:.4f}")
        return metrics

    def visualize_results(self):
        """生成并保存所有可视化图表（每张图单独保存）"""
        sns.set_palette("viridis")

        # 1. PCA 2D聚类
        plt.figure(figsize=(10, 6))
        plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], c=self.kmeans_labels, alpha=0.7)
        plt.title('PCA - K-Means Clustering (2D)')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.colorbar(label='Cluster')
        self._save_plot("pca_2d_kmeans")

        # 2. PCA 3D聚类
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.X_pca[:, 0], self.X_pca[:, 1], self.X_pca[:, 2],
                   c=self.kmeans_labels, alpha=0.7)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('PCA - K-Means (3D)')
        self._save_plot("pca_3d_kmeans", fig)

        # 3. UMAP-DBSCAN聚类
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(self.X_umap[:, 0], self.X_umap[:, 1],
                              c=self.dbscan_labels, alpha=0.7, cmap="rainbow")
        plt.title('UMAP - DBSCAN Clustering')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.colorbar(scatter, label='Cluster')
        noise_ratio = np.sum(self.dbscan_labels == -1) / len(self.dbscan_labels)
        plt.text(0.05, 0.95, f'Noise: {noise_ratio:.1%}',
                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.7))
        self._save_plot("umap_dbscan")

        # 4. 肘部法则
        plt.figure(figsize=(10, 6))
        distortions = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=42).fit(self.X_pca)
            distortions.append(kmeans.inertia_)
        plt.plot(range(1, 11), distortions, 'bx-')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method for Optimal K')
        self._save_plot("elbow_method")

        # 5. 特征相关性热图（仅显示前20个特征）
        plt.figure(figsize=(12, 8))
        n_features = min(20, self.features.shape[1])
        corr = self.features.iloc[:, :n_features].corr()
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt=".2f")
        plt.title(f'Top {n_features} Features Correlation')
        plt.xticks(rotation=45)
        self._save_plot("feature_correlation")

    def _save_plot(self, plot_name, fig=None):
        """保存当前图表到文件"""
        if fig is None:
            fig = plt.gcf()
        output_path = os.path.join(self.output_dir, f"{plot_name}_{self.current_time}.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"图表已保存: {output_path}")

    def save_cluster_results(self):
        """保存聚类结果到Excel"""
        results_df = pd.DataFrame({
            'ID': self.ids,
            'KMeans_Cluster': self.kmeans_labels,
            'Hierarchical_Cluster': self.hierarchical_labels,
            'DBSCAN_Cluster': self.dbscan_labels
        })
        # 添加降维结果
        pca_cols = [f'PC{i + 1}' for i in range(self.X_pca.shape[1])]
        results_df[pca_cols] = self.X_pca
        results_df[['UMAP1', 'UMAP2']] = self.X_umap

        output_path = os.path.join(self.output_dir, f"cluster_results_{self.current_time}.xlsx")
        results_df.to_excel(output_path, index=False)
        print(f"聚类结果已保存: {output_path}")


def main():
    # 替换为您的Excel文件路径
    file_path = "PCA2.xlsx"

    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：文件 {file_path} 不存在！")
        return

    analyzer = PeptideAnalyzer(file_path)

    # 执行完整分析流程
    steps = [
        analyzer.load_data,
        analyzer.preprocess_data,
        analyzer.apply_dimensionality_reduction,
        analyzer.apply_clustering,
        analyzer.evaluate_clusters,
        analyzer.visualize_results,
        analyzer.save_cluster_results
    ]

    for step in steps:
        if not step():
            print("分析流程中断！")
            break


if __name__ == "__main__":
    main()