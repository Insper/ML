# 2026.2 — Overview

Welcome to the 2026.2 offering of **Machine Learning**.

## Meetings

| :octicons-location-24: | :fontawesome-regular-calendar: | :fontawesome-regular-clock: |
|-|:-:|:-:|
| Class | Mon. | 09h45 :fontawesome-solid-arrow-right-long: 09h45 |
| Class | Fri. | 09h45 :fontawesome-solid-arrow-right-long: 09h45 |
| Office Hours | Mon. | 14h45 :fontawesome-solid-arrow-right-long: 16h15 |


## Instructors

<div class="grid cards" markdown>

-   :material-account-tie:{ .lg .middle } **Instructor**

    ---

    **Humberto Sandmann**

    [:material-web:](https://hsandmann.github.io/){:target="_blank"}
    [:simple-github:](https://github.com/hsandmann){:target="_blank"}
    [:material-linkedin:](https://www.linkedin.com/in/hsandmann/){:target="_blank"}

-   :material-account-tie:{ .lg .middle } **Assistant Instructor**

    ---

    **Luciano Pinheiro**

    [:simple-github:](https://github.com/lucianopinheirob){:target="_blank"}
    [:material-linkedin:](https://www.linkedin.com/in/lucianopinheirob/){:target="_blank"}

</div>

## Students

## Grade

The final grade is define by:

$$
\text{Final} = \left\{\begin{array}{lll}
    \text{IG} \geq 5 \bigwedge \text{TG} \geq 5 &
    \implies &
    \displaystyle \text{IG} \times 0.6 + \text{TG} \times 0.4
    \\
    \\
    \text{Otherwise} &
    \implies &
    \min\left(\text{IG}, \text{TG}\right)
    \end{array}\right.
$$

<div class="grade-wrap" markdown>

<div class="grade-box" markdown>

<p class="grade-box-title">Individual</p>

| Component | When | Weight |
|---|---|--:|
| <span class='calendar-mexam'>Midterm Exam</span> | Sep 17–23 | 40% |
| <span class='calendar-fexam'>Final Exam</span> | Nov 5–11 | 60% |

</div>

<div class="grade-box" markdown>

<p class="grade-box-title">Team</p>

| Component | When | Weight |
|---|---|--:|
| <span class='calendar-eda'>EDA</span> | Sep 14 | 30% |
| <span class='calendar-classification'>Classification</span> $\lor$ <span class='calendar-regression'>Regression</span> | Oct 26 | 70% |

</div>

</div>

The <span class='calendar-sexam'>Substitutive Exam</span> (Dec 10–11) covers the whole syllabus; eligibility and which exam it replaces follow Insper's academic regulation.

!!! tip "Quizzes"
    Every lesson ends with an interactive quiz. Quizzes are **not graded** — they give immediate feedback and are the closest rehearsal available for exam questions.

!!! danger "ORAL EXAMINATION"

    **ALL GRADES** in this course are subject to an oral examination for validation. If the outcome of the oral examination is negative, that specific grade will be **ZEROED**.

## Calendar

<div class="calendar-wrap" markdown>

<div class="table-fit" markdown>

| | Sun | Mon | Tue | Wed | Thu | Fri | Sat |
|-|-|-|-|-|-|-|-|
| Aug |    | 10 |    |    |    | 14 |    |
|     |    | 17 |    |    |    | 21 |    |
|     |    | 24 |    |    |    | 28 |    |
|     |    | 31 |    |    |    |    |    |
| Sep |    |    |    |    |    | 04 |    |
|     |    |    |    |    |    | 11 |    |
|     |    | <span class='calendar-eda'>14</span> |    |    | <span class='calendar-mexamw'>17</span> | <span class='calendar-mexamw'>18</span> |    |
|     |    | <span class='calendar-mexam'>21</span> | <span class='calendar-mexamw'>22</span> | <span class='calendar-mexamw'>23</span> |    |    |    |
|     |    | 28 |    |    |    |    |    |
| Oct |    |    |    |    |    | 02 |    |
|     |    | 05 |    |    |    | 09 |    |
|     |    |    |    |    |    | 16 |    |
|     |    | 19 |    |    |    | 23 |    |
|     |    | <span class='calendar-classification'>2</span><span class='calendar-regression'>6</span> |    |    |    | 30 |    |
| Nov |    |    |    |    | <span class='calendar-fexamw'>05</span> | <span class='calendar-fexam'>06</span> |    |
|     |    | <span class='calendar-fexam'>09</span> | <span class='calendar-fexamw'>10</span> | <span class='calendar-fexamw'>11</span> |    |    |    |
|     |    | <span class='calendar-ssprint'>16</span> |    |    |    |    |    |
|     |    | 23 |    |    |    | 27 |    |
|     |    | 30 |    |    |    |    |    |
| Dec |    |    |    |    |    | <span class='calendar-fsprint'>04</span> |    |
|     |    |    |    |    | <span class='calendar-sexam'>10</span> | <span class='calendar-sexam'>11</span> |    |

</div>

<div class="calendar-legend" markdown>

<div class="calendar-legend-box" markdown>

<p class="calendar-legend-title">Individual</p>

<span class='calendar-mexam'>Midterm Exam</span>
<span class='calendar-fexam'>Final Exam</span>
<span class='calendar-sexam'>Substitutive Exam</span>

</div>

<div class="calendar-legend-box" markdown>

<p class="calendar-legend-title">Team</p>

<span class='calendar-eda'>EDA</span>
<span class='calendar-classification'>Classification</span>
<span class='calendar-regression'>Regression</span>

</div>

<div class="calendar-legend-box" markdown>

<p class="calendar-legend-title">Sprint</p>

<span class='calendar-ssprint'>Start</span>
<span class='calendar-fsprint'>Exam</span>

</div>

</div>

</div>

## Class materials

Most lessons include a **hands-on Colab notebook** used in class (linked at the end of each lesson page, under *Class materials*). The complete collection — notebooks, slides, datasets, and papers — lives in the [course Drive folder](https://drive.google.com/drive/folders/1lyndgoY0AG64AYwreUjy-Xasxj3Z8Sow){:target="_blank"}.

## Syllabus

<div class="syllabus" markdown>

### Foundations & History

| | | |
|---|---------|--------|
| 1 | [Introduction & History](classes/introduction/index.md) | What is ML, a timeline from least squares (1805) to foundation models |
| 2 | [The ML Landscape](classes/ml-landscape/index.md) | Learning paradigms, the ML workflow, generalization, ethics |

### Working with Data

| | | |
|---|---------|--------|
| 3 | [Exploratory Data Analysis](classes/eda/index.md) | Summary statistics, distributions, correlation, visualization |
| 4 | [Data Preprocessing](classes/preprocessing/index.md) | Scaling, normalization, encoding, missing values, outliers |
| 5 | [Pipelines](classes/pipelines/index.md) | scikit-learn `Pipeline`, `ColumnTransformer`, reproducibility |

### Unsupervised Learning & Text

| | | |
|---|---------|--------|
| 6 | [Dimensionality Reduction](classes/dimensionality-reduction/index.md) | PCA, t-SNE, UMAP |
| 7 | [Clustering](classes/clustering/index.md) | k-means, hierarchical, DBSCAN/HDBSCAN, silhouette |
| 8 | [Text Representation](classes/text-representation/index.md) | Bag-of-words, TF-IDF, n-grams, embeddings |
| 9 | [Topic Modeling & BERTopic](classes/topic-modeling-bertopic/index.md) | LDA, BERTopic: embeddings + UMAP + HDBSCAN + c-TF-IDF |

### Regression & Model Evaluation

| | | |
|---|---------|--------|
| 10 | [Linear Regression](classes/linear-regression/index.md) | Least squares, OLS derivation, assumptions, regression metrics |
| 11 | [Gradient Descent & Regularization](classes/gradient-descent-regularization/index.md) | Batch/stochastic GD, polynomial features, Ridge, Lasso |
| 12 | [Validation & Data Leakage](classes/validation/index.md) | Train/test split, cross-validation, leakage patterns |
| 13 | [Model Selection](classes/model-selection/index.md) | Bias–variance, regression to the mean, `GridSearchCV` |

### Classification

| | | |
|---|---------|--------|
| 14 | [Classification & Metrics](classes/classification-metrics/index.md) | Confusion matrix, accuracy, precision, recall, F1 |
| 15 | [k-Nearest Neighbors](classes/knn/index.md) | Distance metrics, choosing k, curse of dimensionality |
| 16 | [ROC-AUC & Imbalanced Data](classes/roc-imbalanced/index.md) | ROC/PR curves, resampling, SMOTE, class weights |
| 17 | [Logistic Regression](classes/logistic-regression/index.md) | Sigmoid, cross-entropy, gradient descent, regularization |
| 18 | [Support Vector Machines](classes/svm/index.md) | Margins, soft margin, kernel trick, implementation sketch |

### Trees & Ensembles

| | | |
|---|---------|--------|
| 19 | [Decision Trees](classes/decision-trees/index.md) | Entropy, Gini, CART, pruning |
| 20 | [Random Forest](classes/random-forest/index.md) | Bootstrap, bagging, feature importance, out-of-bag error |
| 21 | [Gradient Boosting](classes/gradient-boosting/index.md) | Boosting, GBM, XGBoost, LightGBM |

### Edge Approaches

| | | |
|---|---------|--------|
| 22 | [Neural Networks](classes/neural-networks/index.md) | Perceptron to MLP, backpropagation, bridge to deep learning |
| 23 | [Explainability](classes/explainability/index.md) | Permutation importance, SHAP, LIME |
| 24 | [AutoML](classes/automl/index.md) | Hyperparameter optimization, Optuna, successive halving |
| 25 | [MLOps](classes/mlops/index.md) | Serving, monitoring, drift, reproducibility |
| 26 | [The Frontier](classes/frontier/index.md) | Foundation models, transfer learning, LLMs, what's next |

</div>

