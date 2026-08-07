import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.datasets import make_blobs, make_circles
from sklearn.svm import SVC

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))

# Left: linear SVM, maximum margin and support vectors
X1, y1 = make_blobs(n_samples=60, centers=2, cluster_std=1.0, random_state=6)
clf1 = SVC(kernel='linear', C=1000).fit(X1, y1)

ax = axes[0]
ax.scatter(X1[y1 == 0, 0], X1[y1 == 0, 1], c='#f0883e', s=18)
ax.scatter(X1[y1 == 1, 0], X1[y1 == 1, 1], c='#58a6ff', s=18)
xx, yy = np.meshgrid(np.linspace(*ax.get_xlim(), 100), np.linspace(*ax.get_ylim(), 100))
Z = clf1.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors='gray',
           linestyles=['--', '-', '--'], linewidths=1)
ax.scatter(clf1.support_vectors_[:, 0], clf1.support_vectors_[:, 1],
           s=110, facecolors='none', edgecolors='black', linewidths=1.2,
           label='support vectors')
ax.set_title('Linear SVM: maximum margin', fontsize=10)
ax.legend(fontsize=8, loc='lower right')
ax.set_xticks([]); ax.set_yticks([])

# Right: RBF kernel on circles
X2, y2 = make_circles(n_samples=200, factor=0.35, noise=0.1, random_state=1)
clf2 = SVC(kernel='rbf', C=1, gamma=2).fit(X2, y2)

ax = axes[1]
xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 200), np.linspace(-1.5, 1.5, 200))
Z = clf2.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contourf(xx, yy, Z > 0, cmap=plt.cm.coolwarm, alpha=0.15)
ax.contour(xx, yy, Z, levels=[0], colors='gray', linewidths=1)
ax.scatter(X2[y2 == 0, 0], X2[y2 == 0, 1], c='#f0883e', s=12)
ax.scatter(X2[y2 == 1, 0], X2[y2 == 1, 1], c='#58a6ff', s=12)
ax.set_title('RBF kernel: nonlinear boundary', fontsize=10)
ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
