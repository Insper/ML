# Decision Trees

A decision tree classifies by asking a sequence of simple questions — *petal length ≤ 2.45? income > 5,000?* — walking from root to leaf. Formalized in the 1980s (CART: Breiman et al., 1984; ID3/C4.5: Quinlan, 1986/1993), trees read like flowcharts a domain expert can audit, handle mixed feature types without scaling, and are the **building block of the ensembles** ([random forests](../random-forest/index.md), [gradient boosting](../gradient-boosting/index.md)) that dominate tabular ML today.

``` python exec="on" html="on"
--8<-- "docs/2026.2/classes/decision-trees/tree-plot.py"
```

A depth-2 tree on iris: two thresholds on petal measurements already separate the species almost perfectly — and you can read *why* directly from the picture.

## How a tree is grown

Trees are built **greedily, top-down** (CART): at each node, try every feature and every threshold, and pick the split that makes the two children **purest**; recurse until a stopping rule fires.

### Measuring impurity

For a node with class proportions \(p_1, \dots, p_k\):

**Gini impurity** (CART's default) — the probability that two random draws from the node disagree:

\[
G = 1 - \sum_{c=1}^{k} p_c^2
\]

**Entropy** (ID3 family) — information-theoretic uncertainty:

\[
H = -\sum_{c=1}^{k} p_c \log_2 p_c
\]

Both are 0 for a pure node and maximal for a 50/50 mix; in practice they choose nearly identical splits (Gini is slightly cheaper — no logarithm).

A candidate split \(S\) of node \(N\) into children \(L, R\) is scored by **impurity decrease** (with entropy, called *information gain*):

\[
\Delta = I(N) - \frac{n_L}{n} I(L) - \frac{n_R}{n} I(R)
\]

For **regression trees**, impurity is simply the variance (MSE) of the target in the node, and each leaf predicts the mean of its samples.

```text
GROW(node):
    if stopping rule (depth, min samples, purity): make leaf
    for each feature j, each threshold t:
        score split x_j ≤ t by impurity decrease Δ
    apply best split; GROW(left); GROW(right)
```

Greedy means **no lookahead**: the tree never reconsiders a split that would pay off two levels later (XOR-like patterns can defeat it). Ensembles compensate.

## Overfitting: the tree's chronic disease

Grown without limits, a tree keeps splitting until leaves are pure — happily isolating every noisy point in its own leaf. Trees are **low-bias, high-variance** learners: tiny changes in data can produce a completely different tree.

``` python exec="on" html="on"
--8<-- "docs/2026.2/classes/decision-trees/tree-overfit.py"
```

The unlimited tree (left) carves rectangular islands around individual noise points; `max_depth=4` (right) captures the real structure. Note the axis-aligned, "staircase" boundaries — trees split one feature at a time.

**Controlling complexity** (all are [bias–variance knobs](../model-selection/index.md#the-biasvariance-trade-off) for [cross-validation](../validation/index.md#cross-validation)):

- *Pre-pruning*: `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_impurity_decrease`;
- *Post-pruning*: grow fully, then cut back branches that don't earn their complexity — **cost-complexity pruning** minimizes \(\text{error} + \alpha \cdot \#\text{leaves}\) (`ccp_alpha`), the tree version of [regularization](../gradient-descent-regularization/index.md#regularization).

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=0)
tree.fit(X_train, y_train)          # no scaling needed!
tree.feature_importances_           # impurity-based importances (sum to 1)
```

## Practical profile

| | |
|---|---|
| **Strengths** | interpretable/auditable; no scaling or one-hot for ordinals needed; mixed feature types; captures interactions and nonlinearity natively; fast prediction |
| **Weaknesses** | high variance (unstable); greedy myopia; axis-aligned bias; poor extrapolation (regression predicts constants outside training range) |
| **Reach for it when** | interpretability is the requirement — otherwise use its ensemble descendants |

!!! tip "One tree, rarely; many trees, constantly"
    A single tree trades too much accuracy for its readability. Its true importance is as the **weak learner** inside random forests and gradient boosting — the next two lessons. Understand splits, impurity, and pruning here, and both ensembles become transparent.

---

## Quiz

<div id="quiz-decision-trees"></div>
<script>
buildQuiz('decision-trees', 'Decision Trees', [
  {
    q: "At each node, CART chooses the split that...",
    opts: [
      "maximizes the depth of the tree",
      "maximizes the impurity decrease (weighted purity gain of the children)",
      "separates the two most distant points",
      "minimizes the number of features used"
    ],
    ans: 1,
    exp: "Every (feature, threshold) candidate is scored by Δ = I(parent) − weighted I(children), using Gini or entropy. The best is applied and the process recurses — greedily, with no lookahead."
  },
  {
    q: "A node contains 50% class A and 50% class B. Its Gini impurity is...",
    opts: [
      "0 — the node is balanced",
      "0.5 — the maximum for two classes",
      "1.0",
      "0.25"
    ],
    ans: 1,
    exp: "G = 1 − (0.5² + 0.5²) = 0.5, the worst case for binary: two random draws disagree half the time. A pure node has G = 0."
  },
  {
    q: "Why do unpruned decision trees overfit so readily?",
    opts: [
      "They underfit, not overfit",
      "Grown to purity, they keep splitting until every noisy point gets its own leaf — memorizing the sample (low bias, high variance)",
      "Because they require feature scaling",
      "Because Gini is a biased estimator"
    ],
    ans: 1,
    exp: "Nothing stops the greedy recursion before pure leaves. The resulting model has near-zero training error and unstable, noise-driven structure. Depth limits, leaf minimums, or cost-complexity pruning restore balance."
  },
  {
    q: "Which statement about feature scaling for decision trees is correct?",
    opts: [
      "Standardization is mandatory, as for k-NN",
      "Scaling is unnecessary: splits are thresholds on one feature at a time, unaffected by monotonic transformations",
      "Only min-max scaling works with trees",
      "Trees require all features in [0, 1]"
    ],
    ans: 1,
    exp: "The question 'income ≤ 5000?' partitions the data identically whether income is in reais or standardized units. This is a major practical convenience of tree-based models."
  },
  {
    q: "Cost-complexity pruning (ccp_alpha) minimizes error + α·(number of leaves). Increasing α...",
    opts: [
      "grows a deeper tree",
      "prunes more aggressively, trading training fit for simplicity — the tree analogue of regularization",
      "changes the impurity measure",
      "only affects prediction speed"
    ],
    ans: 1,
    exp: "α prices each leaf. Higher α makes complex trees expensive, cutting branches whose error reduction doesn't justify their cost — directly analogous to λ in Ridge/Lasso."
  },
  {
    q: "A decision-tree regressor trained on houses of 40–200 m² must predict for a 400 m² mansion. It will...",
    opts: [
      "extrapolate the price trend linearly",
      "predict the constant value of the leaf where 400 m² falls — trees cannot extrapolate beyond the training range",
      "refuse to predict",
      "predict the training mean exactly"
    ],
    ans: 1,
    exp: "Leaves predict constants (the mean of their training samples). Beyond the observed range, every input lands in an edge leaf: the prediction plateaus. Linear models extrapolate; trees do not."
  }
]);
</script>
