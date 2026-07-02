import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

rng = np.random.default_rng(3)
x = rng.uniform(0, 10, 40)
y = 2.5 + 1.8 * x + rng.normal(0, 2.2, 40)

b, a = np.polyfit(x, y, 1)
xs = np.linspace(0, 10, 50)
y_hat = a + b * x

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

axes[0].scatter(x, y, s=14, color='#f0883e', label='data')
axes[0].plot(xs, a + b * xs, color='#58a6ff', label=f'ŷ = {a:.2f} + {b:.2f}x')
for xi, yi, yh in zip(x, y, y_hat):
    axes[0].plot([xi, xi], [yi, yh], color='gray', linewidth=0.6, alpha=0.6)
axes[0].set_title('OLS fit — residuals as gray segments', fontsize=9)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].legend(fontsize=8)

axes[1].scatter(y_hat, y - y_hat, s=14, color='#f0883e')
axes[1].axhline(0, color='#58a6ff', linewidth=1)
axes[1].set_title('Residual plot — no pattern = good sign', fontsize=9)
axes[1].set_xlabel('predicted ŷ')
axes[1].set_ylabel('residual y − ŷ')

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
