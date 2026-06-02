import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, matthews_corrcoef,
    cohen_kappa_score, log_loss,
    precision_recall_curve, roc_curve,
    classification_report
)
from sklearn.calibration import calibration_curve


class ClassificationEvaluator:
    def __init__(self, y_true, y_pred, y_proba=None, class_names=None, output_dir="results"):
        """
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_proba: 预测概率 (二分类需要shape=[n_samples, 2])
            class_names: 类别名称列表
            output_dir: 结果输出目录
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.is_binary = len(np.unique(y_true)) == 2
        self.output_dir = output_dir

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化日志文件
        self.log_file = os.path.join(self.output_dir, f"eval_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        with open(self.log_file, 'w') as f:
            f.write("=== 模型评估报告 ===\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"样本数量: {len(y_true)}\n")
            f.write(f"正负样本比例: {sum(y_true)}:{len(y_true) - sum(y_true)}\n\n")

    def _save_figure(self, fig, name):
        """保存单张图片"""
        path = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        self._log(f"已保存图片: {path}")

    def _log(self, message):
        """写入日志文件"""
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")
        print(message)

    def generate_report(self):
        """生成包含所有指标的字典"""
        metrics = {
            'Accuracy': accuracy_score(self.y_true, self.y_pred),
            'Balanced Accuracy': balanced_accuracy_score(self.y_true, self.y_pred),
            'Precision (Macro)': precision_score(self.y_true, self.y_pred, average='macro'),
            'Recall (Macro)': recall_score(self.y_true, self.y_pred, average='macro'),
            'F1 (Macro)': f1_score(self.y_true, self.y_pred, average='macro'),
            'Matthews Correlation': matthews_corrcoef(self.y_true, self.y_pred),
            "Cohen's Kappa": cohen_kappa_score(self.y_true, self.y_pred),
        }

        if self.is_binary and self.y_proba is not None:
            metrics.update({
                'ROC AUC': roc_auc_score(self.y_true, self.y_proba[:, 1]),
                'Average Precision': average_precision_score(self.y_true, self.y_proba[:, 1]),
                'Log Loss': log_loss(self.y_true, self.y_proba)
            })

        # 写入日志文件
        with open(self.log_file, 'a') as f:
            f.write("\n=== 数值指标 ===\n")
            for k, v in metrics.items():
                f.write(f"{k:<25}: {v:.4f}\n")

            f.write("\n=== 详细分类报告 ===\n")
            f.write(classification_report(self.y_true, self.y_pred, target_names=self.class_names))

        return metrics

    def plot_all(self):
        """生成并保存所有可视化图表"""
        # 1. 混淆矩阵
        fig = plt.figure(figsize=(8, 6))
        self._plot_confusion_matrix()
        self._save_figure(fig, "confusion_matrix")

        # 2. ROC曲线（二分类）
        if self.is_binary and self.y_proba is not None:
            fig = plt.figure(figsize=(8, 6))
            self._plot_roc_curve()
            self._save_figure(fig, "roc_curve")

        # 3. 精确率-召回率曲线
        if self.is_binary and self.y_proba is not None:
            fig = plt.figure(figsize=(8, 6))
            self._plot_pr_curve()
            self._save_figure(fig, "precision_recall_curve")

        # 4. 概率校准曲线
        if self.is_binary and self.y_proba is not None:
            fig = plt.figure(figsize=(8, 6))
            self._plot_calibration_curve()
            self._save_figure(fig, "calibration_curve")

        # 5. 指标对比图
        fig = plt.figure(figsize=(8, 6))
        self._plot_metric_comparison()
        self._save_figure(fig, "metric_comparison")

        self._log("\n评估完成！所有结果已保存至以下目录:")
        self._log(f"图片: {os.path.abspath(self.output_dir)}/*.png")
        self._log(f"日志: {os.path.abspath(self.log_file)}")

    def _plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        blue_rgb = np.array([171, 187, 221]) / 255.0  # 关键：除以255归一化
        # 创建深蓝到浅蓝渐变
        colors = [
            [1.0, 1.0, 1.0],# 白
            np.clip(blue_rgb * 2, 0, 1),
            blue_rgb,  # 原始蓝# 中间浅蓝

        ]
        blue_cmap = LinearSegmentedColormap.from_list("blue_gradient", colors)
        sns.heatmap(cm, annot=True, fmt='d',
                    annot_kws={
                        "weight": "bold",  # 加粗
                        "size": 16  # 字体大小
                    },
                    cmap=blue_cmap,
                    xticklabels=self.class_names,
                    yticklabels=self.class_names,)
        plt.yticks(fontsize=12, weight='bold')  # weight='bold' 表示加粗
        plt.xticks(fontsize=12, weight='bold')  # weight='bold' 表示加粗
        plt.ylabel('True Label', fontweight='bold', fontsize=16)
        plt.xlabel('Predicted Label', fontweight='bold', fontsize=16)

    def _plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba[:, 1])
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(self.y_true, self.y_proba[:, 1]):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.tick_params(axis='both', which='major', labelsize=12)  # 主刻度
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate',fontweight='bold', fontsize=16)
        plt.ylabel('True Positive Rate',fontweight='bold', fontsize=16)
        plt.legend(loc="lower right")

    def _plot_pr_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba[:, 1])
        plt.plot(recall, precision,
                 label=f'PR Curve (AP = {average_precision_score(self.y_true, self.y_proba[:, 1]):.2f})')
        plt.tick_params(axis='both', which='major', labelsize=12)  # 主刻度
        plt.xlabel('Recall',fontweight='bold', fontsize=16)
        plt.ylabel('Precision',fontweight='bold', fontsize=16)
        plt.legend(loc="upper right")

    def _plot_calibration_curve(self):
        prob_true, prob_pred = calibration_curve(self.y_true, self.y_proba[:, 1], n_bins=10)
        plt.plot(prob_pred, prob_true, 's-', label='Model')
        plt.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
        plt.tick_params(axis='both', which='major', labelsize=12)  # 主刻度
        plt.ylabel('Fraction of positives',fontweight='bold', fontsize=16)
        plt.xlabel('Mean predicted probability',fontweight='bold', fontsize=16)

        plt.legend()

    def _plot_metric_comparison(self):
        """新增的指标对比图方法"""
        report = self.generate_report()
        if not report:
            raise ValueError("没有可用的指标数据")

        # 转换为DataFrame
        metrics = pd.DataFrame(list(report.items()), columns=['Metric', 'Value'])

        # 过滤非数值指标
        exclude = ['Confusion Matrix', 'Support']
        metrics = metrics[~metrics['Metric'].isin(exclude)]

        # 绘制图表
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics.plot(x='Metric', y='Value', kind='bar', ax=ax, legend=False)

        # 图表装饰
        ax.set_title('Model Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    plt.rcParams['font.family'] = 'Arial'