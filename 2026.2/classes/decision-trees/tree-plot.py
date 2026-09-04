import matplotlib.pyplot as plt
from io import StringIO
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

iris = load_iris()
X = iris.data[:, 2:]  # petal length, petal width
clf = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, iris.target)

fig, ax = plt.subplots(figsize=(8, 4))
plot_tree(clf,
          feature_names=['petal length', 'petal width'],
          class_names=list(iris.target_names),
          filled=True, rounded=True, fontsize=8, ax=ax, impurity=True)

plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())

plt.close()
