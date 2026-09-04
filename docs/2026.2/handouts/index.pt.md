# Handouts

Roteiros de aula completos, em formato de página única ou em notebook para rodar. Cada um traz
todos os gráficos renderizados a partir dos dados reais, o código para reproduzi-los e exercícios
com solução.

São **complementares** ao conteúdo das aulas, não substitutos: aprofundam um tópico específico e servem tanto para acompanhar a aula quanto para consulta posterior.

<div class="grid cards" markdown>

-   :material-chart-scatter-plot:{ .lg .middle } **Análise Exploratória de Dados**

    ---

    **California Housing** — 20.640 distritos do Censo de 1990.

    Auditoria de integridade, univariada numérica e categórica, os três cruzamentos
    (numérica × numérica, categórica × numérica, categórica × categórica), análise multivariada e
    geoespacial. Inclui simulador interativo de assimetria e curtose.

    ~3 h · 19 figuras · 10 exercícios

    [:material-file-document-outline: abrir](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/handout-eda-california-housing.html){:target="_blank"}
    · [:material-school-outline: aula](../classes/eda/index.md)

-   :material-tune-variant:{ .lg .middle } **Normalização e Transformação**

    ---

    **Titanic** — 891 passageiros, 866 valores ausentes.

    Dados faltantes e mecanismos MCAR / MAR / MNAR, diagrama de decisão de técnicas por tipo de
    variável, escalonamento, transformação de forma, codificação de categóricas e vazamento medido
    em quatro cenários.

    ~3 h · 10 figuras · exercício em 7 checkpoints

    [:material-file-document-outline: abrir](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/handout-normalizacao-titanic.html){:target="_blank"}
    · [:material-school-outline: aula](../classes/preprocessing/index.md)

-   :material-chart-bubble:{ .lg .middle } **Redução de Dimensionalidade**

    ---

    **Iris e MNIST** — de 4 a 784 dimensões.

    PCA por duas derivações, escolha de k, reconstrução e interpretação. Onde o PCA falha em
    variedades não lineares, e então t-SNE e UMAP: o que cada um preserva, o que cada um distorce,
    e uma regra de decisão para escolher entre eles. Anexo sobre a maldição da dimensionalidade.

    ~3 h · 12 simuladores ao vivo · lab do Iris em 8 passos + 5 tarefas do MNIST

    [:material-file-document-outline: abrir](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/handout-reducao-dimensionalidade.html){:target="_blank"}
    · [:material-school-outline: aula](../classes/dimensionality-reduction/index.md)

-   :material-notebook-outline:{ .lg .middle } **Do EDA à Redução de Dimensionalidade**

    ---

    **Palmer Penguins** — 344 pinguins, 6 colunas de entrada, 19 ausentes de duas causas.

    Laboratório em aula que percorre o caminho inteiro: univariada, depois bivariada, separação
    treino/teste *antes* de qualquer transformação, pipeline que trata ausentes e escalas sem
    vazar, e os mesmos dados projetados por PCA, t-SNE e UMAP — comparados sob um único critério
    e aplicados a pontos novos.

    Notebook no Colab · 8 etapas, cada uma com valor de verificação

    [:material-google: abrir no Colab](https://colab.research.google.com/drive/1ytcn6p5Jbk-0swDvGmLOV0qnP5oP0fa5?usp=sharing){:target="_blank"}
    · [:material-school-outline: aula](../classes/dimensionality-reduction/index.md)

-   :material-blur:{ .lg .middle } **Clustering**

    ---

    **Iris e dados sintéticos** — `make_blobs`, `make_moons` e uma linha de base aleatória.

    Por que agrupar é escolher, e não descobrir; o algoritmo e a inércia que ele persegue; e a
    silhueta ponto a ponto — as duas métricas feitas à mão sobre oito pontos, antes de qualquer
    biblioteca. Depois, escolher k com as duas juntas, em cinco etapas de laboratório.

    50 min de aula + 70 min de lab · 12 perguntas · Colab com 3 exercícios

    [:material-file-document-outline: roteiro](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/handout-clustering-kmeans.html){:target="_blank"}
    · [:material-google: Colab](https://colab.research.google.com/drive/1RhDCMPwv1cKCXBUUTblJx7dhLm0xP6fj?usp=sharing){:target="_blank"}
    · [:material-school-outline: aula](../classes/clustering/index.md)

    ---

    :material-trophy-outline: **Extra — desafio**

    **178 vinhos**, autoguiado e com o gabarito retido até o fim: cinco escalas, quatro índices
    mais a estatística gap implementada do zero, estabilidade, GMM com BIC e agrupamento no espaço
    reduzido — e então auditar cada decisão contra o gabarito.

    [:material-file-document-outline: abrir o desafio](https://htmlpreview.github.io/?https://raw.githubusercontent.com/hsandmann/biblio/refs/heads/main/material/handouts/lab-clustering-vinhos.html){:target="_blank"}

</div>

!!! tip "Uso em sala"
    Cada roteiro tem índice lateral com âncoras, botão de copiar em todos os blocos de código e
    folha de estilo própria para impressão — o menu, os botões e os elementos interativos são
    omitidos ao imprimir ou gerar PDF.
