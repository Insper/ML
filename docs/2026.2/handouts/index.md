# Handouts

Full class walkthroughs, as single-page documents and as notebooks to run. Each one renders every
figure from the actual data, ships the code to reproduce it, and includes solved exercises.

They **complement** the class pages rather than replace them: each goes deep on one topic and works
both as a companion during class and as a reference afterwards.

!!! note "Written in Portuguese"
    The handouts are course material for the in-person classes and are written in Brazilian
    Portuguese. The code, the library names and the figures are language-independent.

<div class="grid cards" markdown>

-   :material-chart-scatter-plot:{ .lg .middle } **Exploratory Data Analysis**

    ---

    **California Housing** — 20,640 districts from the 1990 Census.

    Integrity audit, univariate analysis for numeric and categorical variables, all three pairings
    (numeric × numeric, categorical × numeric, categorical × categorical), multivariate and
    geospatial analysis. Includes an interactive skewness/kurtosis simulator.

    ~3 h · 19 figures · 10 exercises

    [:material-file-document-outline: open](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/handout-eda-california-housing.html){:target="_blank"}
    · [:material-school-outline: class](../classes/eda/index.md)

-   :material-tune-variant:{ .lg .middle } **Normalization and Transformation**

    ---

    **Titanic** — 891 passengers, 866 missing values.

    Missing data and the MCAR / MAR / MNAR mechanisms, a decision diagram of techniques by variable
    type, scaling, shape transformation, categorical encoding, and leakage measured across four
    scenarios.

    ~3 h · 10 figures · seven-checkpoint exercise

    [:material-file-document-outline: open](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/handout-normalizacao-titanic.html){:target="_blank"}
    · [:material-school-outline: class](../classes/preprocessing/index.md)

-   :material-chart-bubble:{ .lg .middle } **Dimensionality Reduction**

    ---

    **Iris and MNIST** — from 4 dimensions to 784.

    PCA from two derivations, how to choose k, reconstruction and interpretation. Where PCA fails
    on non-linear manifolds, then t-SNE and UMAP: what each one preserves, what each one distorts,
    and a decision rule for choosing between them. Appendix on the curse of dimensionality.

    ~3 h · 12 live simulators · eight-step Iris lab + 5 MNIST tasks

    [:material-file-document-outline: open](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/handout-reducao-dimensionalidade.html){:target="_blank"}
    · [:material-school-outline: class](../classes/dimensionality-reduction/index.md)

-   :material-notebook-outline:{ .lg .middle } **From EDA to Dimensionality Reduction**

    ---

    **Palmer Penguins** — 344 penguins, 6 input columns, 19 missing values from two causes.

    An in-class lab that runs the whole path end to end: univariate then bivariate analysis, a
    train/test split *before* any transformation, a pipeline that handles missing values and
    scaling without leaking, and the same data projected by PCA, t-SNE and UMAP — compared under
    a single criterion, then applied to unseen points.

    Colab notebook · 8 steps, each ending in a check value

    [:material-google: open in Colab](https://colab.research.google.com/drive/1ytcn6p5Jbk-0swDvGmLOV0qnP5oP0fa5?usp=sharing){:target="_blank"}
    · [:material-school-outline: class](../classes/dimensionality-reduction/index.md)

-   :material-blur:{ .lg .middle } **Clustering**

    ---

    **Iris and synthetic data** — `make_blobs`, `make_moons` and a random baseline.

    Why grouping is a choice rather than a discovery, the algorithm and the inertia it chases, and
    silhouette point by point — both metrics worked by hand on eight points before any library.
    Then choosing k with the two of them together, across five lab steps.

    50 min class + 70 min lab · 12 questions · Colab with 3 exercises

    [:material-file-document-outline: handout](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/handout-clustering-kmeans.html){:target="_blank"}
    · [:material-google: Colab](https://colab.research.google.com/drive/1RhDCMPwv1cKCXBUUTblJx7dhLm0xP6fj?usp=sharing){:target="_blank"}
    · [:material-school-outline: class](../classes/clustering/index.md)

    ---

    :material-trophy-outline: **Extra — challenge**

    **178 wines**, self-guided with the answer key held back to the end: five scalings, four
    indices plus a gap statistic written from scratch, stability, GMM with BIC, and clustering in
    reduced space — then audit every decision against the key.

    [:material-file-document-outline: open the challenge](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/lab-clustering-vinhos.html){:target="_blank"}

</div>

!!! tip "Classroom use"
    Each handout has an anchored sidebar index, a copy button on every code block, and its own
    print stylesheet — the menu, buttons and interactive elements are dropped when printing or
    exporting to PDF.
