# Introduction & History

Machine Learning (ML) is the field of study that gives computers the ability to **learn from data** without being explicitly programmed for every situation. Instead of writing rules by hand, we write programs that *infer* the rules from examples.

The classic formal definition is due to Tom Mitchell:

!!! quote "Mitchell (1997)"
    A computer program is said to **learn** from experience *E* with respect to some class of tasks *T* and performance measure *P*, if its performance at tasks in *T*, as measured by *P*, improves with experience *E*.

Every project in this course can be described in that vocabulary. A spam filter, for instance:

- **Task \(T\)**: classify e-mails as spam or not spam;
- **Experience \(E\)**: a corpus of e-mails already labeled by humans;
- **Performance \(P\)**: the fraction of e-mails classified correctly (or a more careful metric, as we will see in [Classification & Metrics](../classification-metrics/index.md)).

## Why learn from data?

Hand-written rules break down when:

1. **The rules are unknown.** Nobody can write down the exact rules that distinguish a malignant tumor from a benign one in a raw image.
2. **The rules change.** Spammers adapt; a static filter decays. A learning system can be retrained.
3. **The rules are too many.** Recognizing handwritten digits with `if/else` statements is hopeless — there are too many ways to write a "7".
4. **Personalization is needed.** A recommender must behave differently for every user; learning from each user's history scales, hand-tuning does not.

```mermaid
flowchart LR
    subgraph Traditional programming
        A[Rules] --> C[Program]
        B[Data] --> C
        C --> D[Answers]
    end
    subgraph Machine learning
        E[Data] --> G[Learning algorithm]
        F[Answers / labels] --> G
        G --> H[Model ≈ rules]
    end
```

Machine learning inverts the traditional flow: instead of *rules + data → answers*, we feed *data + answers* to an algorithm and obtain a **model** — an approximation of the rules — which we then use to answer new, unseen cases.

## Why now?

None of the core ideas are new — least squares is from 1805 — but three forces converged in the last two decades to make ML ubiquitous:

- **Data**: the web, sensors, and digitization produced datasets large enough to learn subtle patterns;
- **Compute**: GPUs and cloud computing made it cheap to fit large models;
- **Algorithms & software**: open-source libraries (scikit-learn, XGBoost, PyTorch) turned decades of research into a few lines of code.

## A brief history of machine learning

The history of ML is a 200-year conversation between **statistics** and **computer science**. The timeline below marks the milestones this course will visit — from the historical basis to the current edge.

[timeline left alternate(./docs/2026.2/classes/introduction/timeline.json)]

### Reading the arc

Three lessons from this history shape the design of this course:

1. **The classics never left.** Least squares (1805) is still the first model you should try on a regression problem. Logistic regression (1958) remains a production workhorse. Understanding them deeply is not archaeology — it is engineering.
2. **Hype cycles are real.** The field went through two "AI winters" (roughly 1974–1980 and 1987–1993) when promises outran results. Honest evaluation — the subject of Part III — is the antidote.
3. **Modern methods are compositions of classical ideas.** BERTopic (2022), which we will study in Part II, is literally a pipeline of embeddings + dimensionality reduction + clustering + TF-IDF — every ingredient is a classical technique. Gradient boosting is functional gradient descent on decision trees. Knowing the parts lets you understand — and debug — the whole.

## A learning machine made of matchboxes

Long before GPUs, Martin Gardner's 1962 *Scientific American* column described a machine that **learns to play a game using only matchboxes and colored beads** — no electronics at all. For the mini-game *Hexapawn* (three pawns per side on a 3×3 board), build one matchbox per possible board position, and fill it with beads for each legal move:

1. to move, open the matchbox for the current position and draw a random bead — that's the move;
2. if the machine eventually **loses** the game, remove the bead that led to the losing move (punishment);
3. if it **wins**, the beads stay (or extra copies are added — reward).

After a few dozen games the bad moves have literally been *removed from the boxes*: the machine plays perfectly. This is **reinforcement learning in its purest mechanical form** (Donald Michie built the same idea for tic-tac-toe in 1961 and called it MENACE), and it makes Mitchell's definition tangible: the task \(T\) is playing Hexapawn, the experience \(E\) is games played, the performance \(P\) is the win rate — and learning is nothing more than *adjusting parameters (beads) based on feedback*.

!!! example "Read the original"
    Gardner, M. (1962). *How to build a game-learning machine and then teach it to play and to win.* Scientific American — [PDF in the class folder](https://drive.google.com/file/d/1HDRFEUHFsCTogX_LRlvTZ2QExIEkSIVD/view){:target="_blank"}. We play this game in the first class.

## Where this course fits

This course covers **classical machine learning** end to end and finishes at the frontier: neural networks, explainability, AutoML, MLOps, and foundation models. Deep learning gets one bridging lecture here; its full treatment lives in the companion course [Artificial Neural Networks and Deep Learning](https://insper.github.io/ann-dl/).

## Class materials

!!! example "Class slides (in Portuguese)"
    - **Aula 01 — Apresentação**: [:simple-googleslides: open the slides](https://docs.google.com/presentation/d/1Rwn_AIJnbEz4Wy6mCiFWs9dVvPlvuN2pxZa3kAhQH-8/edit){:target="_blank"}
    - Hexapawn worksheets: [training](https://drive.google.com/file/d/15-8gOXzlKVyyIR_ZcZkh372UegPcKcbq/view){:target="_blank"} / [test](https://drive.google.com/file/d/1LOY131bEJbEkcFVuJTgtv-7VDyRgVqst/view){:target="_blank"}

---

## Quiz

<div id="quiz-introduction"></div>
<script>
buildQuiz('introduction', 'Introduction & History', [
  {
    q: "According to Mitchell's definition, a program learns when...",
    opts: [
      "it stores more data over time",
      "its performance at a task, measured by some metric, improves with experience",
      "it can pass the Turing Test",
      "it replaces hand-written rules with a database of examples"
    ],
    ans: 1,
    exp: "Mitchell's definition has three ingredients: a task T, a performance measure P, and experience E. Learning means P improves on T as E grows. Storing data or passing the Turing Test are neither necessary nor sufficient."
  },
  {
    q: "In a spam filter described in Mitchell's vocabulary, the corpus of e-mails already labeled by humans is...",
    opts: [
      "the task T",
      "the performance measure P",
      "the experience E",
      "the model"
    ],
    ans: 2,
    exp: "The labeled corpus is what the program learns from — the experience E. The task T is classifying e-mails; the performance P is a metric such as accuracy."
  },
  {
    q: "Which scenario is a poor fit for machine learning compared to hand-written rules?",
    opts: [
      "Recognizing handwritten digits in images",
      "Validating that a credit card number has 16 digits",
      "Detecting spam that adapts over time",
      "Recommending movies personalized to each user"
    ],
    ans: 1,
    exp: "A fixed, fully-known, simple rule (16 digits) should just be programmed directly. ML pays off when rules are unknown, changing, too numerous, or need personalization."
  },
  {
    q: "The method of least squares — still the foundation of linear regression — was first published in...",
    opts: [
      "1805 by Legendre",
      "1936 by Fisher",
      "1958 by Rosenblatt",
      "1986 by Rumelhart, Hinton and Williams"
    ],
    ans: 0,
    exp: "Legendre published least squares in 1805 (Gauss claimed earlier use). Fisher 1936 is linear discriminant analysis, Rosenblatt 1958 is the perceptron, and 1986 is the backpropagation paper."
  },
  {
    q: "What primarily caused the two AI winters?",
    opts: [
      "Lack of any theoretical progress in the field",
      "Promises and expectations that outran actual results, leading to funding cuts",
      "The invention of the support vector machine",
      "Hardware becoming too expensive to manufacture"
    ],
    ans: 1,
    exp: "Both winters (roughly 1974-1980 and 1987-1993) followed cycles of hype: bold claims, disappointing results, then funding collapse — e.g., after the Lighthill Report (1973) and the fall of expert systems."
  },
  {
    q: "Which combination of forces explains why ML became ubiquitous in the last two decades, even though its core ideas are old?",
    opts: [
      "New mathematics, better keyboards, and faster internet",
      "Large datasets, cheap compute (GPUs/cloud), and open-source software",
      "Government regulation, patents, and proprietary data formats",
      "The replacement of statistics by computer science"
    ],
    ans: 1,
    exp: "Data, compute, and algorithms-turned-into-software are the standard three-way explanation. The math of least squares, decision trees, and even neural networks predates the boom by decades."
  }
]);
</script>
