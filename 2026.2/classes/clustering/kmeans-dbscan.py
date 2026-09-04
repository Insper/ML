import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs, make_moons

blobs_X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.9, random_state=7)
moons_X, _ = make_moons(n_samples=300, noise=0.07, random_state=7)

fig, axes = plt.subplots(2, 2, figsize=(8, 6))

configs = [
    (axes[0, 0], blobs_X, KMeans(n_clusters=3, n_init='auto', random_state=0), 'k-means on blobs ✓'),
    (axes[0, 1], moons_X, KMeans(n_clusters=2, n_init='auto', random_state=0), 'k-means on moons ✗'),
    (axes[1, 0], blobs_X, DBSCAN(eps=0.9, min_samples=5), 'DBSCAN on blobs ✓'),
    (axes[1, 1], moons_X, DBSCAN(eps=0.2, min_samples=5), 'DBSCAN on moons ✓'),
]

for ax, X, model, title in configs:
    labels = model.fit_predict(X)
    noise = labels == -1
    ax.scatter(X[~noise, 0], X[~noise, 1], c=labels[~noise], cmap='tab10', s=8)
    if noise.any():
        ax.scatter(X[noise, 0], X[noise, 1], c='lightgray', s=8, label='noise')
        ax.legend(fontsize=7, loc='lower right')
    if isinstance(model, KMeans):
        ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                   marker='x', s=80, c='black')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
