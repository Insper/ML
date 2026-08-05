# 2026.2 — Visão geral

Bem-vindo(a) à oferta 2026.2 de **Machine Learning**.

## Encontros

| :octicons-location-24: | :fontawesome-regular-calendar: | :fontawesome-regular-clock: |
|-|:-:|:-:|
| Aula | Seg. | 09h45 :fontawesome-solid-arrow-right-long: 09h45 |
| Aula | Sex. | 09h45 :fontawesome-solid-arrow-right-long: 09h45 |
| Atendimento | Seg. | 14h45 :fontawesome-solid-arrow-right-long: 16h15 |


## Docentes

<div class="grid cards" markdown>

-   :material-account-tie:{ .lg .middle } **Professor**

    ---

    **Humberto Sandmann**

    [:material-web:](https://hsandmann.github.io/){:target="_blank"}
    [:simple-github:](https://github.com/hsandmann){:target="_blank"}
    [:material-linkedin:](https://www.linkedin.com/in/hsandmann/){:target="_blank"}

-   :material-account-tie:{ .lg .middle } **Professor Assistente**

    ---

    **Luciano Pinheiro**

    [:simple-github:](https://github.com/lucianopinheirob){:target="_blank"}
    [:material-linkedin:](https://www.linkedin.com/in/lucianopinheirob/){:target="_blank"}

</div>

## Alunos


## Ementa

### A — Fundamentos & História

| # | Assunto | Tópicos |
|---|---------|---------|
| 1 | [Introdução & História](classes/introduction/index.md) | O que é ML, uma linha do tempo dos mínimos quadrados (1805) aos modelos de fundação |
| 2 | [O panorama de ML](classes/ml-landscape/index.md) | Paradigmas de aprendizado, o fluxo de trabalho de ML, generalização, ética |

### B — Trabalhando com Dados

| # | Assunto | Tópicos |
|---|---------|---------|
| 3 | [Análise Exploratória de Dados](classes/eda/index.md) | Estatísticas descritivas, distribuições, correlação, visualização |
| 4 | [Pré-processamento de Dados](classes/preprocessing/index.md) | Escalonamento, normalização, codificação, valores faltantes, outliers |
| 5 | [Pipelines](classes/pipelines/index.md) | `Pipeline` do scikit-learn, `ColumnTransformer`, reprodutibilidade |

### C — Aprendizado Não Supervisionado & Texto

| # | Assunto | Tópicos |
|---|---------|---------|
| 6 | [Redução de Dimensionalidade](classes/dimensionality-reduction/index.md) | PCA, t-SNE, UMAP |
| 7 | [Agrupamento](classes/clustering/index.md) | k-means, hierárquico, DBSCAN/HDBSCAN, silhueta |
| 8 | [Representação de Texto](classes/text-representation/index.md) | Bag-of-words, TF-IDF, n-gramas, embeddings |
| 9 | [Modelagem de Tópicos & BERTopic](classes/topic-modeling-bertopic/index.md) | LDA, BERTopic: embeddings + UMAP + HDBSCAN + c-TF-IDF |

### D — Regressão & Avaliação de Modelos

| # | Assunto | Tópicos |
|---|---------|---------|
| 10 | [Regressão Linear](classes/linear-regression/index.md) | Mínimos quadrados, dedução de MQO, hipóteses, métricas de regressão |
| 11 | [Gradiente Descendente & Regularização](classes/gradient-descent-regularization/index.md) | GD em lote/estocástico, atributos polinomiais, Ridge, Lasso |
| 12 | [Validação & Vazamento de Dados](classes/validation/index.md) | Divisão treino/teste, validação cruzada, padrões de vazamento |
| 13 | [Seleção de Modelos](classes/model-selection/index.md) | Viés–variância, regressão à média, `GridSearchCV` |

### E — Classificação

| # | Assunto | Tópicos |
|---|---------|---------|
| 14 | [Classificação & Métricas](classes/classification-metrics/index.md) | Matriz de confusão, acurácia, precisão, revocação, F1 |
| 15 | [k-Vizinhos Mais Próximos](classes/knn/index.md) | Métricas de distância, escolha de k, maldição da dimensionalidade |
| 16 | [ROC-AUC & Dados Desbalanceados](classes/roc-imbalanced/index.md) | Curvas ROC/PR, reamostragem, SMOTE, pesos de classe |
| 17 | [Regressão Logística](classes/logistic-regression/index.md) | Sigmoide, entropia cruzada, gradiente descendente, regularização |
| 18 | [Naive Bayes](classes/naive-bayes/index.md) | Teorema de Bayes, independência condicional, filtragem de spam |
| 19 | [Máquinas de Vetores de Suporte](classes/svm/index.md) | Margens, margem suave, kernel trick, esboço de implementação |

### F — Árvores & Ensembles

| # | Assunto | Tópicos |
|---|---------|---------|
| 20 | [Árvores de Decisão](classes/decision-trees/index.md) | Entropia, Gini, CART, poda |
| 21 | [Random Forest](classes/random-forest/index.md) | Bootstrap, bagging, importância de atributos, erro out-of-bag |
| 22 | [Gradient Boosting](classes/gradient-boosting/index.md) | Boosting, GBM, XGBoost, LightGBM |

### G — Abordagens de Fronteira

| # | Assunto | Tópicos |
|---|---------|---------|
| 23 | [Redes Neurais](classes/neural-networks/index.md) | Do perceptron ao MLP, retropropagação, ponte para deep learning |
| 24 | [Explicabilidade](classes/explainability/index.md) | Importância por permutação, SHAP, LIME |
| 25 | [AutoML](classes/automl/index.md) | Otimização de hiperparâmetros, Optuna, successive halving |
| 26 | [MLOps](classes/mlops/index.md) | Serving, monitoramento, drift, reprodutibilidade |
| 27 | [A Fronteira](classes/frontier/index.md) | Modelos de fundação, transfer learning, LLMs, próximos passos |

## Material de aula

A maioria das aulas inclui um **notebook prático no Colab** usado em sala (linkado ao final de cada página de aula, em *Material de aula*). A coleção completa — notebooks, slides, conjuntos de dados e artigos — está na [pasta do Drive do curso](https://drive.google.com/drive/folders/1lyndgoY0AG64AYwreUjy-Xasxj3Z8Sow){:target="_blank"}.

## Avaliação

!!! warning "Provisório"
    Os critérios de avaliação, as datas das provas e as descrições dos projetos serão publicados aqui no início do semestre.
