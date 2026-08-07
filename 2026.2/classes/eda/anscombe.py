import matplotlib.pyplot as plt
import numpy as np
from io import StringIO

# Anscombe's quartet (Anscombe, 1973)
x123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
y2 = [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
y3 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
y4 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]

datasets = [(x123, y1), (x123, y2), (x123, y3), (x4, y4)]

fig, axes = plt.subplots(1, 4, figsize=(10, 2.6), sharey=True)
for i, (ax, (x, y)) in enumerate(zip(axes, datasets), start=1):
    x, y = np.array(x), np.array(y)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(3, 20, 10)
    ax.plot(x, y, 'o', color='#f0883e', markersize=4)
    ax.plot(xs, a + b * xs, '-', color='#58a6ff', linewidth=1.2)
    ax.set_title(f'Dataset {i}', fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
