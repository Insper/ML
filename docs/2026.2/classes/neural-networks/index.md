# Neural Networks

Part VI begins at the edge — and the edge is built from pieces you already own. A neural network is [logistic regressions](../logistic-regression/index.md) stacked and composed, trained by [gradient descent](../gradient-descent-regularization/index.md#gradient-descent), regularized with penalties you know from [Ridge](../gradient-descent-regularization/index.md#regularization). This lesson is the bridge lecture; the full journey — CNNs, transformers, generative models — lives in the companion course [ANN-DL](https://insper.github.io/ann-dl/).

## From one neuron to a network

Rosenblatt's **perceptron** (1958) computes \(\hat{y} = \operatorname{step}(w^\top x + b)\) — a linear classifier. Minsky & Papert (1969) proved a single such unit cannot solve XOR (no line separates it), triggering the [first AI winter](../introduction/index.md#a-brief-history-of-machine-learning). The escape, made trainable by **backpropagation** (Rumelhart, Hinton & Williams, 1986): **compose** neurons in layers.

A **multi-layer perceptron (MLP)** with one hidden layer:

\[
h = \sigma(W_1 x + b_1) \qquad\text{(hidden layer: learned features)}
\]
\[
\hat{y} = \operatorname{softmax}(W_2\, h + b_2) \qquad\text{(a logistic/softmax layer on top)}
\]

Read it in course vocabulary: the output layer is *exactly* multi-class logistic regression — but instead of running on hand-engineered features (the polynomials you built [here](../gradient-descent-regularization/index.md#from-lines-to-curves-polynomial-features)), it runs on **features \(h\) that the network learns for itself**. That is the whole revolution: **feature engineering becomes part of the optimization.**

### Activation functions: the essential nonlinearity

Without \(\sigma\), stacking layers collapses: \(W_2(W_1 x) = (W_2 W_1)x\) — still linear. The nonlinearity between layers is what buys expressive power. Modern default: **ReLU**, \(\max(0, z)\) — cheap and gradient-friendly. The **universal approximation theorem** (Cybenko, 1989; Hornik, 1991): one hidden layer with enough neurons can approximate any continuous function — existence guaranteed; *learning* it efficiently is what depth, data, and optimization tricks are for.

``` python exec="on" html="on"
--8<-- "docs/2026.2/classes/neural-networks/mlp-moons.py"
```

The single neuron draws its one line; sixteen hidden ReLU units learn a curved boundary — no polynomial features supplied, the hidden layer *invented* the representation.

## Training: backpropagation

Training minimizes [cross-entropy](../logistic-regression/index.md#the-loss-cross-entropy) (or MSE) by [mini-batch gradient descent](../gradient-descent-regularization/index.md#batch-stochastic-and-mini-batch). **Backpropagation** computes the gradients: it is the chain rule, applied layer by layer from the loss backwards, reusing intermediate results:

1. **Forward pass** — compute activations layer by layer, caching them;
2. **Backward pass** — propagate \(\partial L / \partial \text{activation}\) from output to input, obtaining every \(\partial L / \partial W_\ell\) in one sweep;
3. **Update** — step all weights: \(W_\ell \mathrel{-}= \eta\, \partial L / \partial W_\ell\).

New complication: the loss surface is **non-convex** — unlike logistic regression, no global-optimum guarantee. In practice, good local minima abound; momentum methods and **Adam** (adaptive learning rates, 2015) navigate reliably.

Regularization, translated: **L2 penalty** (called *weight decay*), **early stopping** (validation-loss version, as in [boosting](../gradient-boosting/index.md#the-regularization-toolkit)), and one genuinely new trick — **dropout** (randomly silence neurons during training), which trains an implicit [ensemble](../random-forest/index.md) of subnetworks.

```python
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

mlp = make_pipeline(StandardScaler(),               # gradient-trained ⇒ scale!
    MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu',
                  alpha=1e-4,                        # L2 penalty
                  early_stopping=True, max_iter=500, random_state=0))
mlp.fit(X_train, y_train)
```

(scikit-learn's MLP is fine for tabular experiments; serious deep learning uses PyTorch/JAX — see ANN-DL.)

## Why depth, and when

Deep networks stack many hidden layers, learning **hierarchies of features** (edges → textures → parts → objects, in vision). Depth pays off when raw inputs are **perceptual** — pixels, audio, text — where good features are unknown and data is plentiful. That is where deep learning crushed the field from 2012 on ([AlexNet](../introduction/index.md#a-brief-history-of-machine-learning)).

For **tabular data**, the honest current answer remains: [gradient boosting](../gradient-boosting/index.md#forest-or-boosting) usually wins, with less tuning and less data. Choose by data type, not by hype:

| Data | First choice |
|------|--------------|
| tabular / structured | boosted trees ([Part V](../gradient-boosting/index.md)) |
| images, audio, video | CNNs / vision transformers → ANN-DL |
| text | transformers ([embeddings](../text-representation/index.md) you already used) |
| tiny datasets | linear models, [Naive Bayes](../naive-bayes/index.md), k-NN |

---

## Quiz

<div id="quiz-neural-networks"></div>
<script>
buildQuiz('neural-networks', 'Neural Networks', [
  {
    q: "Why does a multi-layer network need nonlinear activation functions between layers?",
    opts: [
      "To make training faster",
      "Without them the composition of linear layers collapses into a single linear map — no more expressive than logistic regression",
      "To keep the weights positive",
      "Because backpropagation requires the sigmoid specifically"
    ],
    ans: 1,
    exp: "W₂(W₁x) = (W₂W₁)x: stacked linear layers are one linear layer. The nonlinearity (ReLU, sigmoid, tanh) between layers is what lets the network build curved decision boundaries and hierarchical features."
  },
  {
    q: "The output layer of a neural classifier (linear scores + softmax + cross-entropy) is exactly...",
    opts: [
      "a decision tree",
      "multi-class logistic regression, operating on features learned by the hidden layers",
      "a support vector machine",
      "principal component analysis"
    ],
    ans: 1,
    exp: "The final layer is the softmax regression from Part IV. The difference is its inputs: hand-engineered features before, learned representations now — feature engineering moved inside the optimization."
  },
  {
    q: "Backpropagation is best described as...",
    opts: [
      "a new optimizer that replaces gradient descent",
      "an efficient application of the chain rule that computes the gradient of the loss with respect to every weight in one backward sweep",
      "a method to initialize weights",
      "a regularization technique"
    ],
    ans: 1,
    exp: "Backprop computes gradients; gradient descent (SGD/Adam) then uses them to update weights. Its efficiency — reusing cached forward activations layer by layer — is what makes training deep networks feasible."
  },
  {
    q: "Unlike logistic regression, training a neural network is a non-convex problem. In practice this means...",
    opts: [
      "neural networks cannot be trained",
      "there is no global-optimum guarantee — but good local minima are typically abundant and reachable with SGD/Adam",
      "the learning rate must be zero",
      "the loss always increases"
    ],
    ans: 1,
    exp: "Convexity (one global minimum) is lost when layers compose. Empirically, over-parameterized networks have many good minima; adaptive optimizers, momentum, and initialization schemes find them reliably."
  },
  {
    q: "Dropout regularizes a network by...",
    opts: [
      "removing the smallest weights permanently",
      "randomly deactivating neurons during training, forcing redundancy — like training an implicit ensemble of subnetworks",
      "reducing the learning rate over time",
      "deleting training examples"
    ],
    ans: 1,
    exp: "Each batch trains a random subnetwork, so no neuron can over-rely on specific others. At test time all neurons are active (with scaling) — effectively averaging the ensemble, echoing the random-forest idea."
  },
  {
    q: "For a 50,000-row customer table with numeric and categorical columns, current evidence suggests you should first try...",
    opts: [
      "a deep MLP — neural networks are always superior",
      "gradient boosting, since boosted trees remain the strongest and cheapest performers on tabular data",
      "a convolutional network",
      "k-means"
    ],
    ans: 1,
    exp: "Deep learning dominates perception (images, audio, text) where features must be learned from raw signal. On structured tables, benchmarks keep favoring XGBoost/LightGBM — with far less tuning and data hunger."
  }
]);
</script>
