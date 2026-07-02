# AutoML

[Grid search](../model-selection/index.md#grid-search-with-cross-validation) works, but it is brute force: cost multiplies per hyperparameter, most of the grid is wasted in bad regions, and someone still has to choose the grid. **AutoML** automates the search — over hyperparameters, and at the ambitious end, over the entire pipeline. The goal is not replacing the practitioner; it is spending your judgment on [framing, data, and validation design](../ml-landscape/index.md#the-ml-workflow) while the machine grinds the search space.

## Beyond grid: random search

Bergstra & Bengio (2012) made a simple, devastating observation: when only a few hyperparameters really matter (the usual case), a **grid wastes evaluations** — it retests the same few values of the important parameter while marching through irrelevant ones. **Random sampling** covers each dimension's range far better at equal budget:

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint

param_dist = {
    'model__learning_rate': loguniform(1e-3, 0.3),   # continuous, log-scaled
    'model__max_depth': randint(2, 10),
    'model__subsample': (0.6, 0.8, 1.0),
}
search = RandomizedSearchCV(pipe, param_dist, n_iter=50, cv=5,
                            scoring='roc_auc', n_jobs=-1, random_state=0)
```

Random search is the honest baseline every smarter method must beat.

## Bayesian optimization: search that learns

Each CV evaluation is expensive — why ignore what previous trials revealed? **Bayesian optimization** treats the score as an unknown function of the hyperparameters and loops:

1. fit a cheap **surrogate model** to (configuration → score) pairs seen so far;
2. choose the next configuration by an **acquisition function** balancing *exploitation* (near known good points) and *exploration* (uncertain regions);
3. evaluate, add to history, repeat.

**Optuna** (Akiba et al., 2019) is the ecosystem favorite (TPE surrogate by default), with a definitive API — you write the search space as a function:

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
    }
    model = xgb.XGBClassifier(**params, n_estimators=300)
    return cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
study.best_params
```

Because the space is *defined in code*, it can be conditional (tune `gamma` only if `kernel == 'rbf'`) — impossible to express as a grid.

## Successive halving: spend less on losers

Orthogonal trick — most configurations reveal their mediocrity **early** (few boosting rounds, small data subsets). **Successive halving** exploits this: start many configurations on a small budget, keep the best fraction, multiply their budget, repeat — a tournament where finalists get full training. **Hyperband** wraps it with multiple aggressiveness levels; Optuna's **pruners** do the same by killing bad trials mid-CV.

```python
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.model_selection import HalvingRandomSearchCV

search = HalvingRandomSearchCV(pipe, param_dist, factor=3,
                               resource='model__n_estimators',
                               max_resources=1000, cv=5)
```

## Full-pipeline AutoML

The widest scope: search over **imputation × encoding × scaling × feature selection × model family × hyperparameters × ensembling** jointly. Systems: **auto-sklearn** (Bayesian optimization + meta-learning warm starts + auto-ensembling), **TPOT** (evolves [pipelines](../pipelines/index.md) by genetic programming), **FLAML** (cost-aware, frugal), **AutoGluon** (multi-layer stacking of many models; frequently the strongest tabular results), plus commercial platforms.

```python
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='churn', eval_metric='roc_auc') \
                .fit(train_df, presets='best_quality', time_limit=3600)
```

## What AutoML cannot do

The search optimizes exactly what you tell it to. **Garbage validation in, garbage model out — at scale.** Still yours:

- problem framing and the choice of [metric](../classification-metrics/index.md) (optimize F1 vs recall@precision = different products);
- the [validation design](../validation/index.md) — leakage, group structure, time ordering: AutoML will happily exploit any leak *harder* than a human would;
- [data quality and features](../eda/index.md) — an hour of feature engineering routinely beats a day of hyperparameter search;
- [fairness, explainability](../explainability/index.md), and the decision to ship.

!!! tip "Practical default"
    Tabular project, 2020s edition: baseline (`Dummy` + logistic regression) → gradient boosting with defaults → **Optuna, ~100 trials** on the booster → consider AutoGluon if the accuracy is worth the compute. Escalate only while the validation score keeps paying for the effort.

---

## Quiz

<div id="quiz-automl"></div>
<script>
buildQuiz('automl', 'AutoML', [
  {
    q: "Why does random search often beat grid search at the same budget?",
    opts: [
      "Random numbers are lucky",
      "When few hyperparameters matter, a grid re-tests the same few values of the important ones, while random sampling covers each dimension's range densely",
      "Grid search cannot use cross-validation",
      "Random search evaluates more configurations per second"
    ],
    ans: 1,
    exp: "Bergstra & Bengio (2012): with a 3×3 grid, the important parameter gets only 3 distinct values; 9 random points give it 9. Effective dimensionality is low, so per-dimension coverage wins."
  },
  {
    q: "What distinguishes Bayesian optimization from random search?",
    opts: [
      "It requires a GPU",
      "It fits a surrogate model to past trial results and picks the next configuration by balancing exploration and exploitation",
      "It evaluates every configuration twice",
      "It only tunes continuous parameters"
    ],
    ans: 1,
    exp: "Each expensive evaluation informs the next choice via the surrogate + acquisition function. Random search ignores history; Bayesian optimization learns from it — fewer trials to reach good regions."
  },
  {
    q: "Successive halving saves compute by...",
    opts: [
      "halving the dataset permanently",
      "giving many configurations a small budget, then repeatedly promoting only the best fraction to larger budgets",
      "using half the CPU cores",
      "removing half the features"
    ],
    ans: 1,
    exp: "Bad configurations usually look bad early (few iterations/samples). The tournament design spends most compute on promising finalists instead of fully training every candidate."
  },
  {
    q: "You give an AutoML system a dataset with target leakage. It will most likely...",
    opts: [
      "detect and remove the leaking feature",
      "exploit the leak aggressively, reporting a spectacular validation score that will collapse in production",
      "refuse to train",
      "return the same model as without leakage"
    ],
    ans: 1,
    exp: "AutoML optimizes the validation score you defined — harder than a human would. Validation design and leak prevention remain the practitioner's job; automation amplifies both good and bad setups."
  },
  {
    q: "Which decision can NOT be delegated to AutoML?",
    opts: [
      "choosing the learning rate of a booster",
      "choosing which model family to try",
      "choosing the business-appropriate metric to optimize and the validation scheme that mirrors production",
      "choosing the number of trees"
    ],
    ans: 2,
    exp: "Hyperparameters and model choice are exactly what AutoML searches. What to optimize (recall vs precision vs cost) and how to validate honestly encode the business problem — that is the human's contract with the machine."
  },
  {
    q: "In Optuna, a 'pruner' is a mechanism that...",
    opts: [
      "removes features with low importance",
      "stops unpromising trials early, based on their intermediate scores",
      "prunes the branches of decision trees",
      "reduces the search space after each trial"
    ],
    ans: 1,
    exp: "Pruning is successive halving's spirit inside Bayesian optimization: a trial reporting poor intermediate results (e.g., after 2 of 5 CV folds) is killed, freeing budget for better candidates."
  }
]);
</script>
