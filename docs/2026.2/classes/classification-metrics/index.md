# Classification & Metrics

**Classification** is supervised learning with a categorical target: spam/ham, default/repay, disease/healthy, or one of many classes (digit 0–9). Parts IV and V build a portfolio of classifiers; this lesson builds the tool you need *before* any of them — knowing **how to measure whether a classifier is any good**. Choosing the wrong metric is not a detail: it silently optimizes the wrong behavior.

## The confusion matrix

For a binary problem, call one class **positive** (usually the rare/interesting one: fraud, disease) and the other **negative**. Every prediction lands in one of four cells:

|  | Predicted positive | Predicted negative |
|---|---|---|
| **Actually positive** | TP (true positive) | FN (false negative) — *miss* |
| **Actually negative** | FP (false positive) — *false alarm* | TN (true negative) |

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
confusion_matrix(y_test, y_pred)          # rows = truth, cols = prediction
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
```

The two error types usually have **very different costs**: a missed cancer (FN) is not the same as an unnecessary follow-up exam (FP); a blocked legitimate transaction (FP) is not the same as an approved fraud (FN). Metrics exist to weigh them explicitly.

## Accuracy — and why it lies

\[
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
\]

Accuracy answers "what fraction of predictions were right?" — reasonable when classes are balanced and errors cost the same. But with **imbalanced classes** it degenerates. Fraud is 0.5% of transactions? The dumb rule "everything is legitimate" scores **99.5% accuracy** while catching *zero* fraud.

!!! danger "The accuracy paradox"
    On imbalanced problems, high accuracy can describe a useless model. Always compare against the majority-class baseline (`DummyClassifier`), and reach for the metrics below.

## Precision and recall

Both focus on the positive class, answering different questions:

\[
\text{Precision} = \frac{TP}{TP + FP}
\qquad\qquad
\text{Recall} = \frac{TP}{TP + FN}
\]

- **Precision** — *of the cases I flagged, how many were real?* High precision = few false alarms;
- **Recall** (sensitivity) — *of the real cases, how many did I catch?* High recall = few misses.

They pull in opposite directions: flag more aggressively and recall rises while precision falls; flag conservatively and the reverse. Which to prioritize is a **domain decision**:

| Application | Costly error | Prioritize |
|-------------|-------------|-----------|
| Cancer screening | missing a patient (FN) | recall |
| Spam filter | losing a real e-mail (FP) | precision |
| Fraud triage for human review | wasting analyst time (FP) vs missed fraud (FN) | balance — depends on capacity |

### F1: one number when you must have one

The **harmonic mean** of precision and recall:

\[
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

The harmonic mean punishes imbalance: precision 1.0 with recall 0.02 gives \(F_1 \approx 0.04\), not the arithmetic 0.51 — you cannot buy a good F1 by maxing one side. The general \(F_\beta\) weighs recall \(\beta\) times as heavily as precision (\(F_2\) for screening, \(F_{0.5}\) for spam).

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Multi-class

The confusion matrix generalizes to \(k \times k\) — off-diagonal cells reveal *which* classes get confused (useful diagnostics: is the model mixing 4s and 9s?). Per-class precision/recall/F1 are combined by:

- **macro** average — mean over classes, all classes equal (small classes count fully);
- **weighted** average — mean weighted by class frequency;
- **micro** average — compute from pooled counts (equals accuracy for single-label problems).

On imbalanced multi-class data, report **macro-F1**: it exposes failure on rare classes that weighted averages hide.

## Scores, thresholds, and calibration

Most classifiers output a **score or probability**, and the label comes from a **threshold** (default 0.5):

```python
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)       # the threshold is a choice!
```

Moving the threshold trades precision against recall — lower it to catch more positives (recall ↑, precision ↓), raise it for cleaner alarms. The threshold is a **business decision applied after training**, and evaluating a model *across all thresholds* is exactly what ROC and precision–recall curves do — the subject of [ROC-AUC & Imbalanced Data](../roc-imbalanced/index.md).

When the predicted probabilities themselves matter (risk pricing, triage ordering), check **calibration**: among cases predicted "70%", do about 70% turn out positive? (`sklearn.calibration.CalibrationDisplay`).

## Class materials

!!! example "Class notebook (in Portuguese)"
    Hands-on notebook used in class — **Aula 13 — Classificação de Dados**:
    [:simple-googlecolab: open in Colab](https://colab.research.google.com/drive/1Kw9tOXyDIrS3r9dsZr7K9drfT9zBdyGx){:target="_blank"}

---

## Quiz

<div id="quiz-classification-metrics"></div>
<script>
buildQuiz('classification-metrics', 'Classification & Metrics', [
  {
    q: "A disease affects 1% of patients. A model that says 'healthy' for everyone achieves...",
    opts: [
      "99% accuracy and 0% recall for the disease class",
      "99% accuracy and 99% recall",
      "50% accuracy",
      "0% accuracy"
    ],
    ans: 0,
    exp: "It is right on the 99% healthy patients (accuracy 0.99) but catches none of the sick ones (TP = 0 → recall 0). The accuracy paradox: on imbalanced data, accuracy can flatter a useless model."
  },
  {
    q: "Precision answers the question...",
    opts: [
      "Of all the actual positives, how many did the model catch?",
      "Of all the cases the model flagged as positive, how many really were?",
      "How many predictions were correct overall?",
      "How well are the probabilities calibrated?"
    ],
    ans: 1,
    exp: "Precision = TP/(TP+FP): the purity of the model's alarms. Recall = TP/(TP+FN) answers the first option — coverage of the real positives."
  },
  {
    q: "For a cancer screening test, which error is usually considered worse, and which metric should therefore be prioritized?",
    opts: [
      "False positives are worse — prioritize precision",
      "False negatives are worse (a missed patient) — prioritize recall",
      "Both are equally bad — use accuracy",
      "Neither — screening does not use classification"
    ],
    ans: 1,
    exp: "A missed cancer can be fatal; a false alarm leads to a follow-up exam. Screening favors recall (sensitivity), accepting more false positives — which downstream confirmatory tests then filter."
  },
  {
    q: "A model has precision 1.00 and recall 0.02. Its F1 is approximately...",
    opts: [
      "0.51, the average of the two",
      "0.04 — the harmonic mean collapses when one component is tiny",
      "1.00, because precision is perfect",
      "0.98"
    ],
    ans: 1,
    exp: "F1 = 2PR/(P+R) = 2·1·0.02/1.02 ≈ 0.039. The harmonic mean is dominated by the weaker component, so you cannot game F1 by maximizing only one side."
  },
  {
    q: "Lowering the decision threshold from 0.5 to 0.2 typically causes...",
    opts: [
      "recall to increase and precision to decrease",
      "precision to increase and recall to decrease",
      "both to increase",
      "no change — the threshold only affects training"
    ],
    ans: 0,
    exp: "A lower bar flags more cases as positive: more true positives get caught (recall ↑) but more negatives get swept in too (precision ↓). The threshold is a post-training business decision."
  },
  {
    q: "On an imbalanced 10-class problem, why report macro-F1 rather than accuracy or weighted-F1?",
    opts: [
      "Macro-F1 is always the largest number",
      "Macro-F1 averages per-class F1 with equal weight, so collapse on rare classes is visible instead of being masked by frequent classes",
      "Weighted-F1 cannot be computed for 10 classes",
      "Accuracy is undefined for multi-class problems"
    ],
    ans: 1,
    exp: "Accuracy and weighted averages are dominated by the big classes; a model can fail every rare class and still look fine. Macro treats each class as equally important, exposing those failures."
  }
]);
</script>
