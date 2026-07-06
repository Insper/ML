import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(data.data, data.target,
                                          test_size=0.3, stratify=data.target,
                                          random_state=0)
rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_tr, y_tr)

imp = permutation_importance(rf, X_te, y_te, n_repeats=15, random_state=0,
                             scoring='roc_auc')
order = imp.importances_mean.argsort()[-8:]

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.boxplot(imp.importances[order].T, orientation='horizontal',
           tick_labels=np.array(data.feature_names)[order],
           patch_artist=True,
           boxprops=dict(facecolor='#d3e5ff'), medianprops=dict(color='#f0883e'))
ax.set_xlabel('drop in ROC-AUC when feature is shuffled (test set)')
ax.set_title('Permutation importance — top 8 features', fontsize=10)
ax.tick_params(labelsize=8)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
