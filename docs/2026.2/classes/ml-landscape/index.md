# The ML Landscape

Before diving into algorithms, we need a map: what kinds of learning exist, what a real ML project looks like from end to end, and what it means for a model to *generalize*.

## Learning paradigms

Machine learning problems are classified by the kind of **feedback** available to the learner.

```mermaid
flowchart TD
    ML[Machine Learning]
    ML --> SUP[Supervised Learning<br><small>labeled examples</small>]
    ML --> UNS[Unsupervised Learning<br><small>no labels</small>]
    ML --> REI[Reinforcement Learning<br><small>rewards from interaction</small>]
    SUP --> REG[Regression<br><small>predict a number</small>]
    SUP --> CLA[Classification<br><small>predict a category</small>]
    UNS --> CLU[Clustering]
    UNS --> DIM[Dimensionality Reduction]
    UNS --> TOP[Topic Modeling]
```

### Supervised learning

The dataset contains inputs \(x\) **and** the desired outputs \(y\) (labels). The goal is to learn a function \(f\) such that \(f(x) \approx y\) on *new* data.

- **Regression** — \(y\) is continuous: predicting house prices, energy demand, a patient's length of stay. (Part III)
- **Classification** — \(y\) is categorical: spam/ham, tumor benign/malignant, which digit is in the image. (Parts IV–V)

### Unsupervised learning

Only inputs \(x\) — no labels. The goal is to discover **structure**:

- **Clustering** — group similar observations (customer segments). ([Clustering](../clustering/index.md))
- **Dimensionality reduction** — compress many features into few informative ones ([PCA, t-SNE, UMAP](../dimensionality-reduction/index.md));
- **Topic modeling** — discover themes in a collection of documents ([BERTopic](../topic-modeling-bertopic/index.md)).

### Reinforcement learning

An **agent** interacts with an environment, receives **rewards**, and learns a policy that maximizes long-term reward — the paradigm behind game-playing systems (AlphaGo) and robotic control. It is out of scope for this course, but you should recognize it on the map.

!!! info "In between"
    Real projects often mix paradigms: **semi-supervised** learning (few labels, many unlabeled examples), **self-supervised** learning (labels manufactured from the data itself — how foundation models are pre-trained), and **weak supervision** (noisy, programmatic labels).

## The ML workflow

A model is a small part of a larger, iterative process. This course is organized around this loop:

```mermaid
flowchart LR
    A[1. Frame the problem] --> B[2. Get & explore data<br><small>EDA — Part I</small>]
    B --> C[3. Prepare data<br><small>preprocessing, pipelines</small>]
    C --> D[4. Train models]
    D --> E[5. Evaluate honestly<br><small>validation — Part III</small>]
    E -->|iterate| C
    E --> F[6. Deploy & monitor<br><small>MLOps — Part VI</small>]
    F -->|data drifts| B
```

Two practical truths about this diagram:

1. **Most of the work is in steps 2–3.** Practitioners routinely report spending the majority of their time understanding and preparing data, not fitting models.
2. **The loop never ends.** Deployed models decay as the world changes (*drift*); monitoring and retraining are part of the job, not an afterthought.

## Generalization: the central problem

A model that memorizes its training data perfectly can still be useless. What matters is performance on **data it has never seen**.

- **Underfitting**: the model is too simple to capture the pattern — poor performance even on training data.
- **Overfitting**: the model captures noise as if it were signal — excellent on training data, poor on new data.

\[
\text{Goal: minimize } \underbrace{\mathbb{E}_{(x,y)\sim \mathcal{D}}\big[L\big(f(x),\,y\big)\big]}_{\text{expected loss on new data}} \quad \text{while only observing a finite sample.}
\]

Everything in Part III — train/test splits, cross-validation, regularization, the bias–variance trade-off — exists to manage this tension. For now, keep one rule:

!!! danger "The golden rule"
    **Never evaluate a model on data it was trained on.** Test data must simulate the future: unseen, untouched, used once.

## No free lunch

The **No Free Lunch theorem** (Wolpert, 1996) says that averaged over *all possible problems*, no learning algorithm is better than any other. In practice this means: there is no universally best model — you must **try several families and validate**. That is why this course teaches a portfolio (linear models, neighbors, kernels, trees, ensembles, networks) rather than a single silver bullet.

## Ethics and responsibility

Models trained on historical data inherit historical bias. Before shipping a model, ask:

- **Fairness** — does the model perform equally across demographic groups? A credit model trained on biased decisions reproduces them at scale.
- **Privacy** — was the data collected with consent? Can individuals be re-identified?
- **Transparency** — can decisions be explained to those affected? (Part VI covers [explainability](../explainability/index.md).)
- **Feedback loops** — does deploying the model change the data it will be retrained on? (Predictive policing is the canonical cautionary tale.)

!!! warning
    "The model said so" is never an acceptable justification for a decision that affects people. You — the practitioner — own the consequences.

---

## Quiz

<div id="quiz-ml-landscape"></div>
<script>
buildQuiz('ml-landscape', 'The ML Landscape', [
  {
    q: "Predicting the selling price of an apartment from its area, location, and age is a problem of...",
    opts: [
      "classification",
      "clustering",
      "regression",
      "reinforcement learning"
    ],
    ans: 2,
    exp: "The target (price) is a continuous number and labels are available, so it is supervised learning of the regression type."
  },
  {
    q: "You have millions of customer purchase records but no labels, and want to discover natural customer segments. Which paradigm fits?",
    opts: [
      "Supervised classification",
      "Unsupervised clustering",
      "Reinforcement learning",
      "Regression"
    ],
    ans: 1,
    exp: "No labels + goal of discovering group structure = clustering, the flagship unsupervised task."
  },
  {
    q: "A model reaches 99.8% accuracy on the data it was trained on but only 62% on new data. This is a classic case of...",
    opts: [
      "underfitting",
      "data drift",
      "overfitting",
      "class imbalance"
    ],
    ans: 2,
    exp: "A large gap between training and unseen-data performance means the model memorized noise in the training set — overfitting. Underfitting would show poor performance on both."
  },
  {
    q: "What does the No Free Lunch theorem imply for practice?",
    opts: [
      "Gradient boosting is always the best choice for tabular data",
      "No algorithm is universally best, so you must compare several model families with honest validation",
      "More data always beats better algorithms",
      "Simple models should always be preferred"
    ],
    ans: 1,
    exp: "Averaged over all possible problems, all algorithms perform the same. On any specific problem, some are much better — but you only find out by validating, which is why we learn a portfolio of methods."
  },
  {
    q: "Which step of the ML workflow typically consumes most of a practitioner's time?",
    opts: [
      "Training the model",
      "Choosing hyperparameters",
      "Understanding and preparing the data",
      "Writing the deployment API"
    ],
    ans: 2,
    exp: "Exploration, cleaning, and preparation dominate real projects — which is why Part I of this course is entirely about data."
  },
  {
    q: "A bank deploys a credit model trained on decades of past loan decisions. What is the main ethical risk?",
    opts: [
      "The model may be too slow for production",
      "Historical bias in past decisions is learned and reproduced at scale",
      "The model may use too much memory",
      "Customers may not know Python"
    ],
    ans: 1,
    exp: "Models learn whatever regularities exist in the training data — including discriminatory patterns in historical decisions. Fairness auditing and explainability are required, not optional."
  }
]);
</script>
