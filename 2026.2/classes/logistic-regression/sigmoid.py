import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

z = np.linspace(-8, 8, 200)
sig = 1 / (1 + np.exp(-z))

fig, axes = plt.subplots(1, 2, figsize=(9, 3))

axes[0].plot(z, sig, color='#58a6ff', linewidth=1.6)
axes[0].axhline(0.5, color='gray', linestyle='--', linewidth=0.8)
axes[0].axvline(0, color='gray', linestyle='--', linewidth=0.8)
axes[0].annotate('decision boundary\n(z = 0 → p = 0.5)', xy=(0, 0.5), xytext=(2, 0.3),
                 fontsize=8, arrowprops=dict(arrowstyle='->', color='gray'))
axes[0].set_xlabel('z = w·x + b')
axes[0].set_ylabel('σ(z)')
axes[0].set_title('Sigmoid: score → probability', fontsize=10)

p = np.linspace(0.001, 0.999, 300)
axes[1].plot(p, -np.log(p), color='#3fb950', label='loss if y = 1')
axes[1].plot(p, -np.log(1 - p), color='#ff7b72', label='loss if y = 0')
axes[1].set_xlabel('predicted probability p̂')
axes[1].set_ylabel('cross-entropy loss')
axes[1].set_ylim(0, 6)
axes[1].set_title('Confidently wrong = heavily punished', fontsize=10)
axes[1].legend(fontsize=8)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
