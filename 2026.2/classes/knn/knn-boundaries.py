import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.neighbors import KNeighborsClassifier

X, y = make_moons(n_samples=200, noise=0.3, random_state=0)

xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 200),
                     np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 200))
grid = np.c_[xx.ravel(), yy.ravel()]

cmap_bg = ListedColormap(['#ffe3cc', '#d3e5ff'])

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for ax, k in zip(axes, [1, 15, 100]):
    clf = KNeighborsClassifier(n_neighbors=k).fit(X, y)
    Z = clf.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cmap_bg, alpha=0.9)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='#f0883e', s=10, edgecolors='none')
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='#58a6ff', s=10, edgecolors='none')
    ax.set_title(f'k = {k}', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
