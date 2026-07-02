import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

rng = np.random.default_rng(42)
# Two features on wildly different scales, with outliers
income = np.concatenate([rng.lognormal(8.5, 0.4, 200), [90000, 120000]])
age = np.concatenate([rng.normal(38, 10, 200), [39, 41]])
X = np.column_stack([income, age])

scalers = [
    ('Raw data', None),
    ('StandardScaler', StandardScaler()),
    ('MinMaxScaler', MinMaxScaler()),
    ('RobustScaler', RobustScaler()),
]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.8))
for ax, (name, scaler) in zip(axes, scalers):
    Z = X if scaler is None else scaler.fit_transform(X)
    ax.scatter(Z[:, 0], Z[:, 1], s=6, color='#f0883e', alpha=0.6)
    ax.set_title(name, fontsize=9)
    ax.set_xlabel('income', fontsize=8)
    if name == 'Raw data':
        ax.set_ylabel('age', fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
