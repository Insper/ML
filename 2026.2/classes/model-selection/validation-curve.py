import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(5)
X = np.sort(rng.uniform(0, 1, 80)).reshape(-1, 1)
y = np.sin(2 * np.pi * X.ravel()) + rng.normal(0, 0.3, 80)

degrees = np.arange(1, 13)
model = make_pipeline(PolynomialFeatures(), StandardScaler(), LinearRegression())
train_scores, val_scores = validation_curve(
    model, X, y,
    param_name='polynomialfeatures__degree', param_range=degrees,
    cv=5, scoring='neg_mean_squared_error',
)

train_mse = -train_scores.mean(axis=1)
val_mse = -val_scores.mean(axis=1)

fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(degrees, train_mse, 'o-', color='#f0883e', markersize=4, label='training error')
ax.plot(degrees, val_mse, 'o-', color='#58a6ff', markersize=4, label='validation error (5-fold CV)')
best = degrees[val_mse.argmin()]
ax.axvline(best, color='gray', linestyle='--', linewidth=1)
ax.annotate(f'sweet spot: degree {best}', xy=(best, val_mse.min()),
            xytext=(best + 1.2, val_mse.min() + 0.15), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='gray'))
ax.set_xlabel('polynomial degree (model complexity)')
ax.set_ylabel('MSE')
ax.set_ylim(0, 1.0)
ax.legend(fontsize=8)
ax.set_title('Underfitting ← complexity → overfitting', fontsize=10)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
