# k-Nearest Neighbors

k-NN (Fix & Hodges, 1951; Cover & Hart, 1967) is the most intuitive classifier in existence: **to classify a new point, look at the \(k\) most similar known points and take a vote.** No equations to fit, no training loop — the "model" *is* the training data.

## The algorithm

To predict for a query point \(x\):

1. compute the distance from \(x\) to every training point;
2. take the \(k\) closest ones;
3. **classification**: predict the majority class among them (optionally weighting closer neighbors more); **regression**: predict their (weighted) average.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

knn = make_pipeline(StandardScaler(),                 # distances need scaling!
                    KNeighborsClassifier(n_neighbors=5))
knn.fit(X_train, y_train)
```

k-NN is a **lazy** (instance-based) learner: `fit` just stores the data. All the work happens at prediction time — the opposite cost profile of most models (slow to predict, instant to "train").

## Distance metrics

The notion of "similar" is a modeling choice. For the Minkowski family,

\[
d_p(a, b) = \Big( \sum_{j=1}^{d} \lvert a_j - b_j \rvert^p \Big)^{1/p}
\]

- \(p = 2\): **Euclidean** — straight-line distance, the default;
- \(p = 1\): **Manhattan** — sum of coordinate differences; less dominated by one large-difference feature;
- **cosine similarity** for text/embedding vectors ([Text Representation](../text-representation/index.md#tf-idf)); **Hamming** for binary vectors.

!!! danger "Scale first — always"
    Distances are dominated by features with large ranges: income (thousands) crushes age (tens). k-NN without [standardization](../preprocessing/index.md#scaling-methods) is a bug, not a model. Likewise, one-hot encode nominal categories — integer-coded categories create [fictitious distances](../preprocessing/index.md#encoding-categorical-features).

## Choosing k: bias–variance in its purest form

\(k\) is the complexity knob, and it maps perfectly onto the [bias–variance trade-off](../model-selection/index.md#the-biasvariance-trade-off) — just inverted (small \(k\) = complex model):

``` python exec="on" html="on"
--8<-- "docs/2026.2/classes/knn/knn-boundaries.py"
```

- **\(k = 1\)**: every training point rules its own island — jagged boundary, noise memorized, training error zero, **high variance** (overfit);
- **\(k = 15\)**: smooth boundary following the true structure — the sweet spot here;
- **\(k = 100\)** (half the dataset): the vote is swamped by the global majority — **high bias** (underfit); at \(k = n\) every prediction is the majority class.

Choose \(k\) by [cross-validation](../validation/index.md#cross-validation); odd values avoid ties in binary problems. Typical good values grow roughly like \(\sqrt{n}\), but validate rather than trust rules of thumb.

## The curse of dimensionality, revisited

k-NN's premise — *near means similar* — degrades as dimensions grow ([Dimensionality Reduction](../dimensionality-reduction/index.md)):

- volume grows exponentially: with uniform data, covering 10% of the samples in \(d=100\) dimensions requires a neighborhood spanning ~98% of each axis — "nearest" neighbors are not near;
- pairwise distances concentrate: the ratio between the farthest and nearest neighbor tends to 1, so the vote becomes arbitrary;
- irrelevant features add pure noise to the distance.

Remedies: feature selection, [PCA/UMAP](../dimensionality-reduction/index.md) before k-NN, or metric learning. Rule of thumb: k-NN shines in **low-to-moderate dimensions with plenty of data**.

## Practical profile

| | |
|---|---|
| **Strengths** | zero training time; naturally multi-class; nonlinear boundaries for free; one intuitive hyperparameter; a strong baseline |
| **Weaknesses** | prediction is \(O(n \cdot d)\) per query (mitigated by KD-trees/ball trees in low dims, approximate NN — FAISS, HNSW — at scale); memory = whole dataset; sensitive to scaling, irrelevant features, and high dimensionality |
| **Classic uses** | recommender candidates ("users like you"), image retrieval, anomaly detection (distance to k-th neighbor), imputation ([KNNImputer](../preprocessing/index.md#missing-value-imputation)), semantic search over embeddings |

The "find the nearest embeddings" operation is also the heart of modern **vector databases** powering retrieval-augmented LLM systems ([The Frontier](../frontier/index.md)) — 1950s ideas serving 2020s systems.

---

## Quiz

<div id="quiz-knn"></div>
<script>
buildQuiz('knn', 'k-Nearest Neighbors', [
  {
    q: "Why is k-NN called a 'lazy' learner?",
    opts: [
      "It converges slowly",
      "fit() merely stores the training data; all computation is deferred to prediction time",
      "It only works on small datasets",
      "It ignores half of the features"
    ],
    ans: 1,
    exp: "There is no training phase — no parameters are estimated. Each prediction searches the stored dataset for neighbors, giving k-NN the inverse cost profile of eager models like linear regression."
  },
  {
    q: "With k = 1, the training error of k-NN is...",
    opts: [
      "always 0 (each training point is its own nearest neighbor) — a red flag for overfitting",
      "always 50%",
      "equal to the test error",
      "undefined"
    ],
    ans: 0,
    exp: "Every training point classifies itself correctly, so training accuracy is trivially perfect while the jagged boundary memorizes noise. Small k = high variance; evaluate on held-out data."
  },
  {
    q: "Setting k equal to the size of the training set makes k-NN...",
    opts: [
      "perfectly accurate",
      "predict the overall majority class for every query — maximal bias",
      "equivalent to k = 1",
      "crash"
    ],
    ans: 1,
    exp: "With all points voting, the local neighborhood is irrelevant: every prediction is the global majority. k controls bias–variance: k=1 overfits, k=n underfits, the sweet spot is found by cross-validation."
  },
  {
    q: "A k-NN model uses raw income (0–500,000) and age (18–90). What happens?",
    opts: [
      "Age dominates because it comes first",
      "Income dominates the Euclidean distance; age becomes nearly irrelevant to the vote",
      "The model balances them automatically",
      "The model refuses to fit"
    ],
    ans: 1,
    exp: "Squared differences in income are millions of times larger than in age. Neighbors are chosen by income alone. Standardization before k-NN is mandatory."
  },
  {
    q: "In very high dimensions, k-NN degrades mainly because...",
    opts: [
      "computers cannot store the data",
      "distances concentrate — the nearest neighbor is barely closer than the farthest — so 'near means similar' breaks down",
      "the majority class disappears",
      "ties become impossible to break"
    ],
    ans: 1,
    exp: "The curse of dimensionality: volume grows exponentially, data becomes sparse, and the contrast between near and far vanishes. Reduce dimensions (PCA/UMAP) or select features before k-NN."
  },
  {
    q: "Which situation plays to k-NN's strengths?",
    opts: [
      "A 10,000-dimensional sparse dataset with 200 samples",
      "Real-time predictions on billions of rows without preprocessing",
      "A low-dimensional, well-scaled dataset with abundant samples and a nonlinear class boundary",
      "A problem requiring interpretable coefficients per feature"
    ],
    ans: 2,
    exp: "Plenty of data in few dimensions makes neighborhoods meaningful, and k-NN fits arbitrary boundaries without assuming a functional form. High dimensions, huge query volumes, or coefficient-style interpretability call for other tools."
  }
]);
</script>
