import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=4000, n_features=10, n_informative=4,
                           weights=[0.95, 0.05], flip_y=0.02, random_state=4)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)

proba = LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

fpr, tpr, _ = roc_curve(y_te, proba)
prec, rec, _ = precision_recall_curve(y_te, proba)
auc = roc_auc_score(y_te, proba)
ap = average_precision_score(y_te, proba)
prevalence = y_te.mean()

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

axes[0].plot(fpr, tpr, color='#58a6ff', label=f'model (AUC = {auc:.2f})')
axes[0].plot([0, 1], [0, 1], '--', color='gray', linewidth=1, label='random (AUC = 0.50)')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate (recall)')
axes[0].set_title('ROC curve', fontsize=10)
axes[0].legend(fontsize=8)

axes[1].plot(rec, prec, color='#f0883e', label=f'model (AP = {ap:.2f})')
axes[1].axhline(prevalence, color='gray', linestyle='--', linewidth=1,
                label=f'random (prevalence = {prevalence:.2f})')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision–Recall curve (5% positives)', fontsize=10)
axes[1].legend(fontsize=8)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
