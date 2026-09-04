import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import StringIO
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df['species'] = df['target'].map(dict(enumerate(iris.target_names)))

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

# Left: distribution of one feature per class
for species, group in df.groupby('species'):
    axes[0].hist(group['petal length (cm)'], bins=15, alpha=0.6, label=species)
axes[0].set_xlabel('petal length (cm)')
axes[0].set_ylabel('count')
axes[0].set_title('Distribution by class', fontsize=10)
axes[0].legend(fontsize=8)

# Right: correlation matrix of the numeric features
corr = df[iris.feature_names].corr()
im = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
axes[1].set_xticks(range(4))
axes[1].set_yticks(range(4))
labels = [c.replace(' (cm)', '') for c in iris.feature_names]
axes[1].set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
axes[1].set_yticklabels(labels, fontsize=7)
axes[1].set_title('Correlation matrix', fontsize=10)
for i in range(4):
    for j in range(4):
        axes[1].text(j, i, f'{corr.iloc[i, j]:.2f}', ha='center', va='center',
                     fontsize=7, color='white' if abs(corr.iloc[i, j]) > 0.6 else 'black')
fig.colorbar(im, ax=axes[1], fraction=0.046)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
