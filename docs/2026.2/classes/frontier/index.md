# The Frontier

The last stop of the arc that began with [least squares in 1805](../introduction/index.md#a-brief-history-of-machine-learning): where the field stands now, how **foundation models** changed the economics of ML, and — just as important for a practitioner — where the classical toolbox you spent this course building **still wins**.

## Transfer learning: stop starting from zero

Classical supervised learning trains each model from scratch on its own labeled data. **Transfer learning** reuses knowledge: pre-train a large model on a huge generic corpus, then adapt it to your task:

- **feature extraction** — freeze the pre-trained model, use its representations as features for a classical head (you did this: [sentence embeddings](../text-representation/index.md#dense-embeddings) + [logistic regression](../logistic-regression/index.md) or [clustering](../topic-modeling-bertopic/index.md));
- **fine-tuning** — continue training some or all weights on your labeled data (typically 10²–10⁴ examples instead of 10⁶).

This inverted the data economics of the field: tasks that once demanded massive labeled datasets became feasible with hundreds of examples.

## Foundation models

Scale the recipe — transformer architectures, **self-supervised pre-training** (predict masked/next tokens: the labels are free, so the entire internet is training data), billions of parameters — and something qualitatively new appears. A **foundation model** (Bommasani et al., 2021) is one giant pre-trained model adapted to many downstream tasks: the GPT family, Claude, Gemini, Llama for text; CLIP and its successors for vision-language; Whisper for speech.

With **large language models**, adaptation gets lighter still:

| Adaptation | Labeled data needed | What happens |
|------------|--------------------:|--------------|
| zero-shot | none | describe the task in the prompt |
| few-shot / in-context | a handful | show examples *in the prompt*; no weights change |
| fine-tuning (full / LoRA) | hundreds+ | update (a low-rank slice of) the weights |
| RAG | none (needs documents) | retrieve relevant passages — via [embedding](../text-representation/index.md) [nearest-neighbor search](../knn/index.md)! — and stuff them into the context |

"Classify this support ticket as billing/technical/other" — in 2018, a labeling project and a trained classifier; today, often a single prompt. **Prompting became the new fit()** for a broad class of language tasks.

## The classical toolbox in an LLM world

Look inside the modern stack and this course is everywhere:

- LLM output layers are [softmax regression](../logistic-regression/index.md#multi-class-softmax); training is [mini-batch gradient descent](../gradient-descent-regularization/index.md#batch-stochastic-and-mini-batch) with [weight decay](../neural-networks/index.md#training-backpropagation) on [cross-entropy](../logistic-regression/index.md#the-loss-cross-entropy);
- RAG systems are [k-NN](../knn/index.md) over [embeddings](../text-representation/index.md) (vector databases = approximate nearest-neighbor engines);
- evaluation of LLM systems is [precision/recall thinking](../classification-metrics/index.md) plus [honest held-out design](../validation/index.md) — contaminated test sets are the field's current leakage scandal;
- and [BERTopic](../topic-modeling-bertopic/index.md) showed foundation-model embeddings composed with UMAP + HDBSCAN + TF-IDF.

### Where classical ML still wins

Reach for Parts I–V, not a foundation model, when:

| Situation | Why classical wins |
|-----------|--------------------|
| **Tabular data** (churn, credit, pricing, demand) | [gradient boosting](../gradient-boosting/index.md#forest-or-boosting) still tops benchmarks; LLMs are poor at tables of numbers |
| **Latency / cost / scale** (ms decisions, billions of rows) | logistic regression serves in microseconds for ~zero cost |
| **Regulated decisions** | [odds ratios](../logistic-regression/index.md#odds-and-interpretability) and [SHAP](../explainability/index.md#shap-local-global) satisfy auditors; a prompt does not |
| **Small, well-structured problems** | a 2,000-row dataset needs [bias control](../model-selection/index.md#the-biasvariance-trade-off), not a trillion parameters |
| **Determinism and stability** | fixed model + fixed input = fixed output; LLMs sample |

The frontier practitioner's skill is **routing**: perception and language → foundation models; structured prediction → the classical stack; systems → both (an LLM parses the free-text complaint; XGBoost scores the churn risk; [MLOps](../mlops/index.md) monitors both).

## Open problems — where this field is going

- **Hallucination and reliability** — fluent falsehoods; calibration ([Metrics](../classification-metrics/index.md#scores-thresholds-and-calibration)) at frontier scale;
- **Evaluation** — benchmarks saturate and leak into training data; honest measurement is an arms race ([Validation](../validation/index.md), scaled up);
- **Alignment and safety** — making systems pursue intended goals; RLHF and successors;
- **Efficiency** — distillation, quantization, small specialized models vs giant generalists;
- **Data provenance, bias, and governance** — the [ethics questions](../ml-landscape/index.md#ethics-and-responsibility), now industrialized;
- **Agents** — models that plan, call tools, and act; evaluation and safety largely unsolved.

!!! quote "The course thesis, one last time"
    Every "new" idea you will meet is a composition of things you now understand: losses, gradients, regularization, embeddings, neighbors, ensembles, honest validation. The frontier moves; the foundations compound. Learn the parts, and no whole will be a black box.

**Continue at**: [Artificial Neural Networks and Deep Learning](https://insper.github.io/ann-dl/) — architectures, transformers, and generative models in full depth.

---

## Quiz

<div id="quiz-frontier"></div>
<script>
buildQuiz('frontier', 'The Frontier', [
  {
    q: "What makes self-supervised pre-training so scalable?",
    opts: [
      "It uses smaller models",
      "The labels are manufactured from the data itself (e.g., predict the next token), so no human annotation limits the training-set size",
      "It skips gradient descent",
      "It only trains on curated datasets"
    ],
    ans: 1,
    exp: "Supervised learning is bottlenecked by labeling. Next-token/masked-token prediction turns any raw text into training pairs for free — enabling training on internet-scale corpora, the fuel of foundation models."
  },
  {
    q: "Few-shot (in-context) learning differs from fine-tuning in that...",
    opts: [
      "it requires more labeled data",
      "the examples are provided in the prompt and no weights are updated",
      "it permanently modifies the model",
      "it only works for images"
    ],
    ans: 1,
    exp: "In-context learning conditions the frozen model on examples at inference time. Fine-tuning (full or LoRA) actually updates parameters and persists. The lighter the adaptation, the cheaper the iteration."
  },
  {
    q: "A RAG (retrieval-augmented generation) system finds relevant documents using...",
    opts: [
      "a decision tree over keywords",
      "nearest-neighbor search over dense embeddings — the k-NN + text-representation machinery of this course",
      "the LLM's internal memory only",
      "random sampling of the corpus"
    ],
    ans: 1,
    exp: "Documents and query are embedded (sentence-transformer style); a vector database performs (approximate) k-NN by cosine similarity; top passages are injected into the prompt. Classical components, frontier application."
  },
  {
    q: "For predicting churn from a structured customer table, the evidence still favors...",
    opts: [
      "prompting a large language model with each row",
      "gradient boosting on well-engineered features, honestly validated",
      "zero-shot classification",
      "training a giant transformer from scratch on the table"
    ],
    ans: 1,
    exp: "LLMs are weak on numeric tabular prediction, expensive at scale, and hard to audit. Boosted trees remain the benchmark leaders there — knowing when NOT to use a foundation model is a frontier skill."
  },
  {
    q: "Benchmark contamination — test questions appearing in a model's training data — is the frontier-scale version of which concept from this course?",
    opts: [
      "the bias-variance trade-off",
      "data leakage: evaluating on data the model has effectively seen inflates scores",
      "regression to the mean",
      "the curse of dimensionality"
    ],
    ans: 1,
    exp: "It is exactly train/test contamination from the validation lesson, at internet scale: the model 'memorized the exam'. Honest evaluation — fresh, held-out, uncontaminated — remains the hardest discipline in ML."
  },
  {
    q: "Which situation genuinely calls for a foundation model rather than the classical stack?",
    opts: [
      "scoring 50 million rows of transactions in a nightly batch under budget",
      "extracting intent and entities from free-text customer messages with few labeled examples",
      "a regulated credit decision needing auditable feature contributions",
      "a 500-row dataset of sensor readings"
    ],
    ans: 1,
    exp: "Unstructured language with scarce labels is exactly where pre-trained knowledge pays. The other three — scale/cost, auditability, tiny structured data — favor the classical toolbox."
  }
]);
</script>
