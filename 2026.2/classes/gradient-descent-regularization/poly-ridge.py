import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(1)
x = np.sort(rng.uniform(0, 1, 25))
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, 25)
xs = np.linspace(0, 1, 200)

models = [
    ('Degree 1 — underfit', make_pipeline(PolynomialFeatures(1), LinearRegression())),
    ('Degree 15 — overfit', make_pipeline(PolynomialFeatures(15), StandardScaler(), LinearRegression())),
    ('Degree 15 + Ridge(α=0.01)', make_pipeline(PolynomialFeatures(15), StandardScaler(), Ridge(alpha=0.01))),
]

fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for ax, (title, model) in zip(axes, models):
    model.fit(x.reshape(-1, 1), y)
    ax.scatter(x, y, s=12, color='#f0883e')
    ax.plot(xs, np.sin(2 * np.pi * xs), '--', color='gray', linewidth=1, label='true function')
    ax.plot(xs, model.predict(xs.reshape(-1, 1)), color='#58a6ff', linewidth=1.4, label='model')
    ax.set_ylim(-1.8, 1.8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
axes[0].legend(fontsize=7)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
