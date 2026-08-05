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

## Nota

A nota final é definida por:

$$
\text{Final} = \left\{\begin{array}{lll}
    \text{NI} \geq 5 \bigwedge \text{NG} \geq 5 &
    \implies &
    \displaystyle \frac{ \text{NI} + \text{NG} } {2}
    \\
    \\
    \text{Caso contrário} &
    \implies &
    \min\left(\text{NI}, \text{NG}\right)
    \end{array}\right.
$$

<div class="grade-wrap" markdown>

<div class="grade-box" markdown>

<p class="grade-box-title">Individual</p>

| Componente | Quando | Peso |
|---|---|---|
| <span class='calendar-mexam'>Avaliação Intermediária</span> | 17–23 set | 40% |
| <span class='calendar-fexam'>Avaliação Final</span> | 5–11 nov | 60% |

</div>

<div class="grade-box" markdown>

<p class="grade-box-title">Grupo</p>

| Componente | Quando | Peso |
|---|---|---|
| <span class='calendar-eda'>EDA</span> | 11 set | 20% |
| <span class='calendar-classification'>Classificação</span> | 26 out | 30% |
| <span class='calendar-regression'>Regressão</span> | 26 out | 30% |

</div>

</div>

A <span class='calendar-sexam'>Avaliação Substitutiva</span> (10–11 dez) cobre toda a ementa; a elegibilidade e qual avaliação ela substitui seguem o regulamento acadêmico do Insper.

!!! tip "Quizzes"
    Toda aula termina com um quiz interativo. Os quizzes **não valem nota** — dão retorno imediato e são o ensaio mais próximo das questões das avaliações.

## Calendário

<div class="calendar-wrap" markdown>

<div class="table-fit" markdown>

| | Dom | Seg | Ter | Qua | Qui | Sex | Sáb |
|-|-|-|-|-|-|-|-|
| Ago |    | 10 |    |    |    | 14 |    |
|     |    | 17 |    |    |    | 21 |    |
|     |    | 24 |    |    |    | 28 |    |
|     |    | 31 |    |    |    |    |    |
| Set |    |    |    |    |    | 04 |    |
|     |    |    |    |    |    | <span class='calendar-eda'>11</span> |    |
|     |    | 14 |    |    | <span class='calendar-mexam'>17</span> | <span class='calendar-mexam'>18</span> |    |
|     |    | <span class='calendar-mexam'>21</span> | <span class='calendar-mexam'>22</span> | <span class='calendar-mexam'>23</span> |    |    |    |
|     |    | 28 |    |    |    |    |    |
| Out |    |    |    |    |    | 02 |    |
|     |    | 05 |    |    |    | 09 |    |
|     |    |    |    |    |    | 16 |    |
|     |    | 19 |    |    |    | 23 |    |
|     |    | <span class='calendar-classification'>2</span><span class='calendar-regression'>6</span> |    |    |    | 30 |    |
| Nov |    |    |    |    | <span class='calendar-fexam'>05</span> | <span class='calendar-fexam'>06</span> |    |
|     |    | <span class='calendar-fexam'>09</span> | <span class='calendar-fexam'>10</span> | <span class='calendar-fexam'>11</span> |    |    |    |
|     |    | <span class='calendar-ssprint'>16</span> |    |    |    |    |    |
|     |    | 23 |    |    |    | 27 |    |
|     |    | 30 |    |    |    |    |    |
| Dez |    |    |    |    |    | <span class='calendar-fsprint'>04</span> |    |
|     |    |    |    |    | <span class='calendar-sexam'>10</span> | <span class='calendar-sexam'>11</span> |    |

</div>

<div class="calendar-legend" markdown>

<div class="calendar-legend-box" markdown>

<p class="calendar-legend-title">Individual</p>

<span class='calendar-mexam'>Avaliação Intermediária</span>
<span class='calendar-fexam'>Avaliação Final</span>
<span class='calendar-sexam'>Avaliação Substitutiva</span>

</div>

<div class="calendar-legend-box" markdown>

<p class="calendar-legend-title">Equipe</p>

<span class='calendar-eda'>EDA</span>
<span class='calendar-classification'>Classificação</span>
<span class='calendar-regression'>Regressão</span>

</div>

<span class='calendar-ssprint'>Início da Sprint</span>
<span class='calendar-fsprint'>Prova da Sprint</span>

</div>

</div>

## Ementa

<div class="syllabus" markdown>

### Fundamentos & História

| | | |
|---|---------|---------|
| 1 | [Introdução & História](classes/introduction/index.md) | O que é ML, uma linha do tempo dos mínimos quadrados (1805) aos modelos de fundação |
| 2 | [O panorama de ML](classes/ml-landscape/index.md) | Paradigmas de aprendizado, o fluxo de trabalho de ML, generalização, ética |

### Trabalhando com Dados

| | | |
|---|---------|---------|
| 3 | [Análise Exploratória de Dados](classes/eda/index.md) | Estatísticas descritivas, distribuições, correlação, visualização |
| 4 | [Pré-processamento de Dados](classes/preprocessing/index.md) | Escalonamento, normalização, codificação, valores faltantes, outliers |
| 5 | [Pipelines](classes/pipelines/index.md) | `Pipeline` do scikit-learn, `ColumnTransformer`, reprodutibilidade |

### Aprendizado Não Supervisionado & Texto

| | | |
|---|---------|---------|
| 6 | [Redução de Dimensionalidade](classes/dimensionality-reduction/index.md) | PCA, t-SNE, UMAP |
| 7 | [Agrupamento](classes/clustering/index.md) | k-means, hierárquico, DBSCAN/HDBSCAN, silhueta |
| 8 | [Representação de Texto](classes/text-representation/index.md) | Bag-of-words, TF-IDF, n-gramas, embeddings |
| 9 | [Modelagem de Tópicos & BERTopic](classes/topic-modeling-bertopic/index.md) | LDA, BERTopic: embeddings + UMAP + HDBSCAN + c-TF-IDF |

### Regressão & Avaliação de Modelos

| | | |
|---|---------|---------|
| 10 | [Regressão Linear](classes/linear-regression/index.md) | Mínimos quadrados, dedução de MQO, hipóteses, métricas de regressão |
| 11 | [Gradiente Descendente & Regularização](classes/gradient-descent-regularization/index.md) | GD em lote/estocástico, atributos polinomiais, Ridge, Lasso |
| 12 | [Validação & Vazamento de Dados](classes/validation/index.md) | Divisão treino/teste, validação cruzada, padrões de vazamento |
| 13 | [Seleção de Modelos](classes/model-selection/index.md) | Viés–variância, regressão à média, `GridSearchCV` |

### Classificação

| | | |
|---|---------|---------|
| 14 | [Classificação & Métricas](classes/classification-metrics/index.md) | Matriz de confusão, acurácia, precisão, revocação, F1 |
| 15 | [k-Vizinhos Mais Próximos](classes/knn/index.md) | Métricas de distância, escolha de k, maldição da dimensionalidade |
| 16 | [ROC-AUC & Dados Desbalanceados](classes/roc-imbalanced/index.md) | Curvas ROC/PR, reamostragem, SMOTE, pesos de classe |
| 17 | [Regressão Logística](classes/logistic-regression/index.md) | Sigmoide, entropia cruzada, gradiente descendente, regularização |
| 18 | [Máquinas de Vetores de Suporte](classes/svm/index.md) | Margens, margem suave, kernel trick, esboço de implementação |

### Árvores & Ensembles

| | | |
|---|---------|---------|
| 19 | [Árvores de Decisão](classes/decision-trees/index.md) | Entropia, Gini, CART, poda |
| 20 | [Random Forest](classes/random-forest/index.md) | Bootstrap, bagging, importância de atributos, erro out-of-bag |
| 21 | [Gradient Boosting](classes/gradient-boosting/index.md) | Boosting, GBM, XGBoost, LightGBM |

### Abordagens de Fronteira

| | | |
|---|---------|---------|
| 22 | [Redes Neurais](classes/neural-networks/index.md) | Do perceptron ao MLP, retropropagação, ponte para deep learning |
| 23 | [Explicabilidade](classes/explainability/index.md) | Importância por permutação, SHAP, LIME |
| 24 | [AutoML](classes/automl/index.md) | Otimização de hiperparâmetros, Optuna, successive halving |
| 25 | [MLOps](classes/mlops/index.md) | Serving, monitoramento, drift, reprodutibilidade |
| 26 | [A Fronteira](classes/frontier/index.md) | Modelos de fundação, transfer learning, LLMs, próximos passos |

</div>

## Material de aula

A maioria das aulas inclui um **notebook prático no Colab** usado em sala (linkado ao final de cada página de aula, em *Material de aula*). A coleção completa — notebooks, slides, conjuntos de dados e artigos — está na [pasta do Drive do curso](https://drive.google.com/drive/folders/1lyndgoY0AG64AYwreUjy-Xasxj3Z8Sow){:target="_blank"}.
