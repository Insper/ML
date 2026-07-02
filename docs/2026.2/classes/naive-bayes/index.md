# Naive Bayes

Naive Bayes is classical ML at its purest: apply an 18th-century theorem (Bayes, 1763) with one bold simplifying assumption, and get a classifier that is fast, needs little data, and filtered your spam for two decades. It also ties Part IV back to [text representation](../text-representation/index.md) — bag-of-words is its natural habitat.

## Bayes' theorem

For a class \(y\) and observed features \(x\):

\[
\underbrace{P(y \mid x)}_{\text{posterior}} = \frac{\overbrace{P(x \mid y)}^{\text{likelihood}}\;\overbrace{P(y)}^{\text{prior}}}{\underbrace{P(x)}_{\text{evidence}}}
\]

Read as a learning rule: start from the **prior** (how common is each class?), weigh by the **likelihood** (how typical are these features for that class?), and obtain the **posterior** (how probable is the class, given what we observed?). Classify by the largest posterior — \(P(x)\) is the same for all classes and cancels:

\[
\hat{y} = \arg\max_y \; P(y)\, P(x \mid y)
\]

## The "naive" assumption

The likelihood \(P(x_1, \dots, x_d \mid y)\) is a joint distribution over all feature combinations — hopeless to estimate. Naive Bayes assumes features are **conditionally independent given the class**:

\[
P(x_1, \dots, x_d \mid y) \;\approx\; \prod_{j=1}^{d} P(x_j \mid y)
\qquad\Longrightarrow\qquad
\hat{y} = \arg\max_y \; P(y) \prod_{j=1}^{d} P(x_j \mid y)
\]

Now each \(P(x_j \mid y)\) is a simple one-dimensional estimate: count frequencies (discrete features) or fit a Gaussian (continuous ones). Training = counting. One pass over the data.

The assumption is almost always **false** (in real spam, "free" and "offer" co-occur far more than independence predicts). Why does it still work? Classification needs only the **ranking** of posteriors, not their exact values. Naive Bayes usually gets \(\arg\max\) right even when the probabilities themselves are badly distorted — typically over-confident (pushed toward 0 or 1). Trust its labels, not its probabilities.

## The classic: spam filtering

With [bag-of-words](../text-representation/index.md#bag-of-words) features, **Multinomial Naive Bayes** models word counts per class. Estimate from training counts:

\[
P(\text{spam}) = \frac{\#\text{spam docs}}{\#\text{docs}},
\qquad
P(w \mid \text{spam}) = \frac{\text{count}(w, \text{spam}) + 1}{\sum_{w'} \text{count}(w', \text{spam}) + |V|}
\]

The \(+1\) is **Laplace smoothing**: without it, a single word never seen in spam training data gives \(P(w \mid \text{spam}) = 0\), and one zero **annihilates the entire product** — any e-mail containing that word could never be spam. Smoothing pretends every word was seen once.

In practice, multiply-many-small-numbers underflows, so implementations sum logs:

\[
\hat{y} = \arg\max_y \Big[ \log P(y) + \sum_j \log P(x_j \mid y) \Big]
\]

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

spam_filter = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))  # alpha = smoothing
spam_filter.fit(docs_train, y_train)
spam_filter.predict(["WIN a FREE prize now!!!"])
```

## The Naive Bayes family

| Variant | Feature type | \(P(x_j \mid y)\) model | Typical use |
|---------|-------------|------------------------|-------------|
| `MultinomialNB` | counts | multinomial over counts | text (bag-of-words / TF-IDF) |
| `BernoulliNB` | binary | Bernoulli (presence/absence) | short text, binary flags |
| `GaussianNB` | continuous | Gaussian per feature per class | numeric tabular data |

For `GaussianNB`, each class simply stores a mean and variance per feature:

\[
P(x_j \mid y) = \frac{1}{\sqrt{2\pi\sigma_{jy}^2}} \exp\!\Big(-\frac{(x_j - \mu_{jy})^2}{2\sigma_{jy}^2}\Big)
\]

## Practical profile

| | |
|---|---|
| **Strengths** | trains in one pass (fastest learner in the course); works with little data; handles very high dimensions (10⁵ word features); naturally multi-class; online-updatable (`partial_fit`); no scaling needed |
| **Weaknesses** | probability estimates poorly calibrated; correlated features double-count evidence; linear-ish decision boundaries; the independence fiction can genuinely hurt |
| **Reach for it when** | text baselines, many features + few samples, latency-critical or streaming prediction |

!!! tip "The perfect baseline"
    Whatever fancy model you plan for a text problem, fit `CountVectorizer + MultinomialNB` first. It takes seconds, and anything that cannot beat it is not worth deploying.

---

## Quiz

<div id="quiz-naive-bayes"></div>
<script>
buildQuiz('naive-bayes', 'Naive Bayes', [
  {
    q: "In Bayes' theorem applied to classification, the prior P(y) represents...",
    opts: [
      "the probability of the features",
      "how common each class is before observing any features",
      "the model's confidence after seeing the data",
      "the smoothing parameter"
    ],
    ans: 1,
    exp: "The prior encodes base rates (e.g., 40% of e-mail is spam). The likelihood P(x|y) then updates it with feature evidence, yielding the posterior P(y|x)."
  },
  {
    q: "The 'naive' assumption states that...",
    opts: [
      "all classes are equally likely",
      "features are conditionally independent given the class, so the joint likelihood factorizes into a product of per-feature terms",
      "features follow a uniform distribution",
      "the training data has no noise"
    ],
    ans: 1,
    exp: "P(x₁,...,x_d|y) ≈ ∏P(x_j|y). This reduces an impossible joint estimation problem to d one-dimensional ones — countable from data in a single pass."
  },
  {
    q: "Why does Naive Bayes often classify well even though its independence assumption is false?",
    opts: [
      "The assumption is actually true for most datasets",
      "Classification only needs the argmax over classes: the ranking of posteriors often survives even badly distorted probability estimates",
      "scikit-learn corrects the assumption internally",
      "It does not — it always performs poorly"
    ],
    ans: 1,
    exp: "Violations distort the posterior magnitudes (typically toward overconfidence) but frequently preserve which class scores highest. Hence: trust the labels, distrust the probabilities."
  },
  {
    q: "Without Laplace smoothing, a word that never appeared in spam training data would...",
    opts: [
      "be ignored by the model",
      "make P(spam | any e-mail containing it) = 0, since one zero factor annihilates the whole product",
      "increase the prior of spam",
      "cause a division by zero in the prior"
    ],
    ans: 1,
    exp: "The likelihood is a product; a single P(w|spam) = 0 zeroes it regardless of all other evidence. Adding 1 to every count (α = 1) guarantees no probability is exactly zero."
  },
  {
    q: "For classifying e-mails represented as word counts, the appropriate variant is...",
    opts: [
      "GaussianNB",
      "MultinomialNB",
      "BernoulliNB with continuous features",
      "LinearRegression"
    ],
    ans: 1,
    exp: "MultinomialNB models count data — the natural match for bag-of-words. GaussianNB is for continuous features; BernoulliNB for binary presence/absence vectors."
  },
  {
    q: "Two features are nearly duplicated in the dataset. For Naive Bayes this causes...",
    opts: [
      "no effect — features are independent",
      "the shared evidence to be counted twice, pushing posteriors toward overconfidence",
      "a training error",
      "automatic removal of one feature"
    ],
    ans: 1,
    exp: "Independence-given-class means every feature multiplies its evidence in. Duplicated (correlated) features inject the same evidence multiple times — a concrete way the naive assumption bites."
  }
]);
</script>
