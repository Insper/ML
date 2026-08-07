import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(2)
X = np.sort(rng.uniform(0, 6, 100)).reshape(-1, 1)
y = np.sin(X.ravel()) + rng.normal(0, 0.25, 100)
xs = np.linspace(0, 6, 300).reshape(-1, 1)

gbm = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1,
                                max_depth=2, random_state=0).fit(X, y)

stages = [1, 5, 50, 300]
preds = {}
for i, pred in enumerate(gbm.staged_predict(xs), start=1):
    if i in stages:
        preds[i] = pred.copy()

fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
for ax, m in zip(axes, stages):
    ax.scatter(X, y, s=8, color='#f0883e', alpha=0.6)
    ax.plot(xs, np.sin(xs.ravel()), '--', color='gray', linewidth=1)
    ax.plot(xs, preds[m], color='#58a6ff', linewidth=1.5)
    ax.set_title(f'{m} tree{"s" if m > 1 else ""}', fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
