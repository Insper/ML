# Linear Regression

Linear regression is the **oldest algorithm in this course** — Legendre published least squares in 1805 to fit comet orbits, Gauss claimed it earlier, and Galton's studies of heredity (1886) gave "regression" its name. Two centuries later, it remains the first model to try on any regression problem: fast, interpretable, and a surprisingly strong baseline.

## The model

Predict a continuous target as a weighted sum of features:

\[
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_d x_d = w_0 + \sum_{j=1}^{d} w_j x_j
\]

- \(w_j\): the change in \(\hat{y}\) per unit change in \(x_j\), *holding the other features fixed*;
- \(w_0\) (intercept/bias): the prediction when all features are zero.

In matrix form, with a leading column of 1s absorbed into \(X\): \(\hat{y} = Xw\).

## Least squares

Choose \(w\) to minimize the **sum of squared residuals** — the squared vertical distances between data and fitted line:

\[
J(w) = \sum_{i=1}^{n} \big(y_i - \hat{y}_i\big)^2 = \lVert y - Xw \rVert^2
\]

Why *squares*? Squaring penalizes large errors heavily, yields a smooth (differentiable) objective, and — under Gaussian noise — coincides with maximum likelihood estimation.

### The closed-form solution

Setting the gradient to zero, \(\nabla_w J = -2X^\top(y - Xw) = 0\), gives the **normal equations**:

\[
X^\top X \, w = X^\top y
\qquad\Longrightarrow\qquad
\hat{w} = (X^\top X)^{-1} X^\top y
\]

For simple (one-feature) regression this reduces to the formulas worth memorizing:

\[
\hat{w}_1 = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sum_i (x_i - \bar{x})^2} = r_{xy}\frac{s_y}{s_x},
\qquad
\hat{w}_0 = \bar{y} - \hat{w}_1 \bar{x}
\]

— the slope is the [correlation](../eda/index.md#correlation-carefully) rescaled by the standard deviations, and the line always passes through \((\bar{x}, \bar{y})\).

``` python exec="on" html="on"
--8<-- "docs/2026.2/classes/linear-regression/ols-fit.py"
```

The **residual plot** (right) is the standard diagnostic: residuals should look like structureless noise around zero. Curvature suggests a missing nonlinear term; a funnel shape suggests non-constant variance; isolated extreme residuals point to outliers (recall [Anscombe's quartet](../eda/index.md#why-statistics-are-not-enough)).

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X_train, y_train)
model.coef_, model.intercept_
y_pred = model.predict(X_test)
```

!!! note "When the closed form struggles"
    Inverting \(X^\top X\) costs \(O(d^3)\) and fails when features are perfectly collinear. For huge or ill-conditioned problems we switch to [gradient descent and regularization](../gradient-descent-regularization/index.md) — the next lesson.

## Assumptions behind the inferences

OLS *predictions* need little; but *interpreting coefficients and error bars* leans on the classical assumptions:

1. **Linearity** — the true relationship is (approximately) linear in the features;
2. **Independence** — residuals are not correlated with each other (beware time series);
3. **Homoscedasticity** — residual variance is constant across the range of \(\hat{y}\);
4. **Normality of residuals** — needed for exact confidence intervals and p-values;
5. **No severe multicollinearity** — highly [correlated features](../eda/index.md) make individual coefficients unstable (their *sum* may be well determined while each one swings wildly).

## Measuring regression quality

With residuals \(e_i = y_i - \hat{y}_i\):

| Metric | Formula | Reading |
|--------|---------|---------|
| MAE | \(\frac{1}{n}\sum \lvert e_i \rvert\) | mean error in target units; robust to outliers |
| MSE | \(\frac{1}{n}\sum e_i^2\) | punishes large errors; the training objective |
| RMSE | \(\sqrt{\text{MSE}}\) | like MSE but back in target units |
| R² | \(1 - \frac{\sum e_i^2}{\sum (y_i - \bar{y})^2}\) | fraction of variance explained; 1 = perfect, 0 = no better than predicting \(\bar{y}\) |

R² can be **negative** on test data — the model is then *worse* than the constant baseline \(\bar{y}\). Always compare against that baseline (`DummyRegressor`): it is embarrassing how often a fancy model barely beats it.

!!! warning "Evaluate on held-out data"
    All metrics above are only meaningful on data the model did not see — the subject of [Validation & Data Leakage](../validation/index.md).

---

## Quiz

<div id="quiz-linear-regression"></div>
<script>
buildQuiz('linear-regression', 'Linear Regression', [
  {
    q: "What does ordinary least squares minimize?",
    opts: [
      "The sum of absolute residuals",
      "The sum of squared vertical distances between observed and predicted values",
      "The number of features used",
      "The correlation between features"
    ],
    ans: 1,
    exp: "OLS minimizes J(w) = Σ(yᵢ − ŷᵢ)². Squaring gives a smooth objective with the closed-form solution ŵ = (XᵀX)⁻¹Xᵀy and corresponds to maximum likelihood under Gaussian noise."
  },
  {
    q: "In a fitted model price = 50000 + 3200·area − 1500·age, the coefficient 3200 means...",
    opts: [
      "the price of an average apartment",
      "each extra unit of area adds 3200 to the predicted price, holding age fixed",
      "area is 3200 times more important than age",
      "the model explains 32% of the variance"
    ],
    ans: 1,
    exp: "A linear coefficient is a partial effect: the predicted change per unit of that feature with others held constant. Comparing raw magnitudes across features only makes sense after standardization."
  },
  {
    q: "Your residual plot shows a clear U-shaped curve. What does this indicate?",
    opts: [
      "The model is perfect",
      "The relationship has a nonlinear component the linear model is missing",
      "The residuals are homoscedastic",
      "The target should be one-hot encoded"
    ],
    ans: 1,
    exp: "Systematic curvature in residuals means the linear form is wrong — consider adding polynomial terms or transforming variables. Good residuals look like patternless noise around zero."
  },
  {
    q: "On the test set your model gets R² = −0.3. The correct interpretation is...",
    opts: [
      "impossible — R² is always between 0 and 1",
      "the model explains 30% of the variance",
      "the model performs worse than simply predicting the mean of y for every observation",
      "the features are negatively correlated with y"
    ],
    ans: 2,
    exp: "R² = 1 − SSE/SST. On held-out data SSE can exceed SST, making R² negative: the constant baseline ŷ = ȳ beats your model. Always check the dummy baseline."
  },
  {
    q: "Two features are almost perfectly correlated (r = 0.99). Fitting OLS with both typically causes...",
    opts: [
      "a syntax error",
      "unstable individual coefficients with huge variance, even though predictions may stay fine",
      "the intercept to disappear",
      "R² to become exactly zero"
    ],
    ans: 1,
    exp: "Multicollinearity makes XᵀX nearly singular: the data cannot tell the two coefficients apart, so they swing wildly (often with opposite signs) while their combined effect — and the predictions — remain stable."
  },
  {
    q: "Why is RMSE often reported instead of MSE?",
    opts: [
      "RMSE is always smaller",
      "RMSE is in the same units as the target, making it interpretable (e.g., 'off by R$ 23k on average')",
      "MSE cannot be computed on test data",
      "RMSE ignores outliers"
    ],
    ans: 1,
    exp: "MSE is in squared units (R$²...), which nobody can read. Its square root returns to target units while preserving the same ranking of models."
  }
]);
</script>
