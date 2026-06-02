import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm, datasets
from sklearn.linear_model import LogisticRegression
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
import graphviz
import xgboost as xgb
from sklearn.tree import export_graphviz

# 1. 学术绘图配置
rcParams.update({
    'font.family': 'Times New Roman',
    'mathtext.fontset': 'stix',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight'
})

# 2. 生成模拟数据
np.random.seed(42)
X, y = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0,
                                    n_classes=2, weights=[0.5, 0.5], class_sep=1.5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# ====================== 3. 随机森林示意图 ======================
plt.figure(figsize=(12, 6))
rf = RandomForestClassifier(n_estimators=3, max_depth=2)
rf.fit(X_train, y_train)

# 绘制前3棵决策树
for i, tree_in_forest in enumerate(rf.estimators_[:3]):
    plt.subplot(1, 3, i + 1)
    plot_tree(tree_in_forest, filled=True, feature_names=['Feature 1', 'Feature 2'],
              class_names=['Class 0', 'Class 1'], rounded=True)
    plt.title(f"Tree {i + 1}", fontsize=10)
plt.suptitle("Random Forest: Individual Decision Trees", y=1.02)
plt.savefig('RF_trees.pdf')

# 导出单棵树的Graphviz图
dot_data = tree.export_graphviz(rf.estimators_[0], out_file=None,
                                feature_names=['Feature 1', 'Feature 2'],
                                class_names=['Class 0', 'Class 1'],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("RF_single_tree", format='pdf')

# ====================== 4. SVM决策边界示意图 ======================
plt.figure(figsize=(8, 6))
svc = svm.SVC(kernel='linear', C=1.0)
svc.fit(X_train, y_train)

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svc.decision_function(xy).reshape(XX.shape)

# 绘制数据点和支持向量
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1],
           s=100, facecolors='none', edgecolors='k', linewidths=1.5)

# 标注超平面和间隔
ax.contour(XX, YY, Z, colors='r', levels=[-1, 0, 1], alpha=0.8,
           linestyles=['--', '-', '--'])
ax.text(0.5, -0.1, r"$w^Tx + b = 0$", transform=ax.transAxes,
        ha='center', color='red', fontsize=12)
plt.title("SVM Decision Boundary with Support Vectors")
plt.savefig('SVM_decision_boundary.pdf')

# ====================== 5. 逻辑回归示意图 ======================
plt.figure(figsize=(12, 5))

# 左侧：Sigmoid函数曲线
plt.subplot(1, 2, 1)
z = np.linspace(-5, 5, 100)
sigma = 1 / (1 + np.exp(-z))
plt.plot(z, sigma, 'b-', linewidth=2)
plt.axvline(0, color='k', linestyle='--')
plt.xlabel("Linear Combination ($w^Tx + b$)")
plt.ylabel("Probability $\sigma(z)$")
plt.title("Sigmoid Activation Function")

# 右侧：决策边界可视化
plt.subplot(1, 2, 2)
lr = LogisticRegression()
lr.fit(X_train, y_train)

coef = lr.coef_[0]
x_boundary = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_boundary = -(coef[0] * x_boundary + lr.intercept_) / coef[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
plt.plot(x_boundary, y_boundary, 'r--', label="Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Logistic Regression Decision Boundary")
plt.legend()
plt.tight_layout()
plt.savefig('LR_visualization.pdf')

# ====================== 6. XGBoost示意图 ======================
plt.figure(figsize=(12, 6))

# 训练XGBoost模型
xgb_clf = xgb.XGBClassifier(n_estimators=3, max_depth=2)
xgb_clf.fit(X_train, y_train)

# 绘制第一棵树的特征重要性
plt.subplot(1, 2, 1)
xgb.plot_importance(xgb_clf, ax=plt.gca(), height=0.5)
plt.title("Feature Importance (Gain)")

# 绘制第一棵树的结构
plt.subplot(1, 2, 2)
xgb.plot_tree(xgb_clf, num_trees=0, ax=plt.gca())
plt.title("First Tree Structure")
plt.tight_layout()
plt.savefig('XGBoost_components.pdf')

# 7. 组合对比图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
models = {
    "Random Forest": rf,
    "SVM": svc,
    "Logistic Regression": lr,
    "XGBoost": xgb_clf
}

for ax, (name, model) in zip(axes.ravel(), models.items()):
    if hasattr(model, "predict_proba"):
        Z = model.predict_proba(np.c_[XX.ravel(), YY.ravel()])[:, 1]
    else:
        Z = model.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)

    ax.contourf(XX, YY, Z, alpha=0.8, cmap=plt.cm.RdBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdBu)
    ax.set_title(name)

plt.suptitle("Model Decision Boundary Comparison", y=1.02)
plt.tight_layout()
plt.savefig('model_comparison.pdf')