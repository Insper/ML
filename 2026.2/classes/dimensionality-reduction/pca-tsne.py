import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

digits = load_digits()
rng = np.random.default_rng(0)
idx = rng.choice(len(digits.data), 500, replace=False)
X = StandardScaler().fit_transform(digits.data[idx])
y = digits.target[idx]

Z_pca = PCA(n_components=2, random_state=0).fit_transform(X)
Z_tsne = TSNE(n_components=2, perplexity=30, random_state=0, init='pca').fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, Z, title in [(axes[0], Z_pca, 'PCA (linear)'), (axes[1], Z_tsne, 't-SNE (nonlinear)')]:
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap='tab10', s=8)
    ax.set_title(f'{title} — digits 0–9', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(sc, ax=axes, fraction=0.025, ticks=range(10))

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True, bbox_inches='tight')
print(buffer.getvalue())

plt.close()
