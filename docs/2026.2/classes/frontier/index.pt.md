# A Fronteira

A última parada do arco que começou com os [mínimos quadrados em 1805](../introduction/index.md#uma-breve-historia-do-machine-learning): onde o campo está agora, como os **modelos de fundação** mudaram a economia do ML e — tão importante para um profissional — onde a caixa de ferramentas clássica que você passou este curso construindo **ainda vence**.

## Transfer learning: pare de começar do zero

O aprendizado supervisionado clássico treina cada modelo do zero com seus próprios dados rotulados. O **transfer learning** reutiliza conhecimento: pré-treine um modelo grande em um corpus genérico enorme e depois o adapte à sua tarefa:

- **extração de atributos** — congelar o modelo pré-treinado e usar suas representações como atributos para uma cabeça clássica (você fez isso: [sentence embeddings](../text-representation/index.md#embeddings-densos) + [regressão logística](../logistic-regression/index.md) ou [agrupamento](../topic-modeling-bertopic/index.md));
- **fine-tuning** — continuar treinando alguns ou todos os pesos com seus dados rotulados (tipicamente 10²–10⁴ exemplos em vez de 10⁶).

Isso inverteu a economia de dados do campo: tarefas que antes exigiam conjuntos rotulados massivos tornaram-se viáveis com centenas de exemplos.

## Modelos de fundação

Escale a receita — arquiteturas transformer, **pré-treino autossupervisionado** (prever tokens mascarados/seguintes: os rótulos são grátis, então a internet inteira vira dado de treino), bilhões de parâmetros — e algo qualitativamente novo aparece. Um **modelo de fundação** (Bommasani et al., 2021) é um único modelo gigante pré-treinado adaptado a muitas tarefas subsequentes: a família GPT, o Claude, o Gemini, o Llama para texto; o CLIP e seus sucessores para visão-linguagem; o Whisper para fala.

Com os **grandes modelos de linguagem (LLMs)**, a adaptação fica ainda mais leve:

| Adaptação | Dados rotulados necessários | O que acontece |
|-----------|----------------------------:|----------------|
| zero-shot | nenhum | descreva a tarefa no prompt |
| few-shot / in-context | um punhado | mostre exemplos *no prompt*; nenhum peso muda |
| fine-tuning (completo / LoRA) | centenas+ | atualiza (uma fatia de baixo posto d)os pesos |
| RAG | nenhum (precisa de documentos) | recupera passagens relevantes — via [busca por vizinhos mais próximos](../knn/index.md) sobre [embeddings](../text-representation/index.md)! — e as enfia no contexto |

"Classifique este tíquete de suporte como cobrança/técnico/outro" — em 2018, um projeto de rotulagem e um classificador treinado; hoje, muitas vezes um único prompt. **O prompting virou o novo fit()** para uma ampla classe de tarefas de linguagem.

## A caixa de ferramentas clássica em um mundo de LLMs

Olhe dentro da pilha moderna e este curso está em toda parte:

- as camadas de saída dos LLMs são [regressão softmax](../logistic-regression/index.md#multiclasse-softmax); o treino é [gradiente descendente em mini-batch](../gradient-descent-regularization/index.md#batch-estocastico-e-mini-batch) com [weight decay](../neural-networks/index.md#treino-retropropagacao) sobre [entropia cruzada](../logistic-regression/index.md#a-perda-entropia-cruzada);
- os sistemas RAG são [k-NN](../knn/index.md) sobre [embeddings](../text-representation/index.md) (bancos de dados vetoriais = motores de vizinhos mais próximos aproximados);
- a avaliação de sistemas de LLM é o [pensamento de precisão/revocação](../classification-metrics/index.md) mais um [desenho honesto de dados separados](../validation/index.md) — conjuntos de teste contaminados são o atual escândalo de vazamento do campo;
- e o [BERTopic](../topic-modeling-bertopic/index.md) mostrou embeddings de modelos de fundação compostos com UMAP + HDBSCAN + TF-IDF.

### Onde o ML clássico ainda vence

Recorra às Partes I–V, não a um modelo de fundação, quando:

| Situação | Por que o clássico vence |
|----------|--------------------------|
| **Dados tabulares** (churn, crédito, precificação, demanda) | o [gradient boosting](../gradient-boosting/index.md#floresta-ou-boosting) ainda lidera os benchmarks; LLMs são ruins com tabelas de números |
| **Latência / custo / escala** (decisões em ms, bilhões de linhas) | a regressão logística serve em microssegundos por ~zero custo |
| **Decisões reguladas** | [razões de chances](../logistic-regression/index.md#chances-odds-e-interpretabilidade) e [SHAP](../explainability/index.md#shap-local-global) satisfazem auditores; um prompt não |
| **Problemas pequenos e bem estruturados** | um conjunto de 2.000 linhas precisa de [controle de viés](../model-selection/index.md#o-trade-off-viesvariancia), não de um trilhão de parâmetros |
| **Determinismo e estabilidade** | modelo fixo + entrada fixa = saída fixa; LLMs amostram |

A habilidade do profissional de fronteira é o **roteamento**: percepção e linguagem → modelos de fundação; predição estruturada → a pilha clássica; sistemas → ambos (um LLM interpreta a reclamação em texto livre; o XGBoost escora o risco de churn; o [MLOps](../mlops/index.md) monitora os dois).

## Problemas em aberto — para onde este campo vai

- **Alucinação e confiabilidade** — falsidades fluentes; calibração ([Métricas](../classification-metrics/index.md#escores-limiares-e-calibracao)) em escala de fronteira;
- **Avaliação** — os benchmarks saturam e vazam para os dados de treino; a medição honesta é uma corrida armamentista ([Validação](../validation/index.md), ampliada);
- **Alinhamento e segurança** — fazer os sistemas perseguirem os objetivos pretendidos; RLHF e sucessores;
- **Eficiência** — destilação, quantização, modelos pequenos especializados vs generalistas gigantes;
- **Proveniência de dados, viés e governança** — as [questões de ética](../ml-landscape/index.md#etica-e-responsabilidade), agora industrializadas;
- **Agentes** — modelos que planejam, chamam ferramentas e agem; avaliação e segurança em grande parte não resolvidas.

!!! quote "A tese do curso, uma última vez"
    Toda ideia "nova" que você encontrará é uma composição de coisas que você agora entende: perdas, gradientes, regularização, embeddings, vizinhos, ensembles, validação honesta. A fronteira se move; os fundamentos se acumulam. Aprenda as partes, e nenhum todo será uma caixa-preta.

**Continue em**: [Artificial Neural Networks and Deep Learning](https://insper.github.io/ann-dl/) — arquiteturas, transformers e modelos generativos em profundidade total.

---

## Quiz

<div id="quiz-frontier"></div>
<script>
buildQuiz('frontier', 'A Fronteira', [
  {
    q: "O que torna o pré-treino autossupervisionado tão escalável?",
    opts: [
      "Ele usa modelos menores",
      "Os rótulos são fabricados a partir dos próprios dados (ex.: prever o próximo token), então nenhuma anotação humana limita o tamanho do conjunto de treino",
      "Ele pula o gradiente descendente",
      "Ele só treina em conjuntos de dados curados"
    ],
    ans: 1,
    exp: "O aprendizado supervisionado é limitado pela rotulagem. A previsão de próximo token/token mascarado transforma qualquer texto bruto em pares de treino de graça — viabilizando o treino em corpora em escala de internet, o combustível dos modelos de fundação."
  },
  {
    q: "O aprendizado few-shot (in-context) difere do fine-tuning em que...",
    opts: [
      "exige mais dados rotulados",
      "os exemplos são fornecidos no prompt e nenhum peso é atualizado",
      "modifica o modelo permanentemente",
      "só funciona para imagens"
    ],
    ans: 1,
    exp: "O aprendizado in-context condiciona o modelo congelado a exemplos no momento da inferência. O fine-tuning (completo ou LoRA) de fato atualiza os parâmetros e persiste. Quanto mais leve a adaptação, mais barata a iteração."
  },
  {
    q: "Um sistema RAG (geração aumentada por recuperação) encontra documentos relevantes usando...",
    opts: [
      "uma árvore de decisão sobre palavras-chave",
      "busca por vizinhos mais próximos sobre embeddings densos — a maquinaria de k-NN + representação de texto deste curso",
      "apenas a memória interna do LLM",
      "amostragem aleatória do corpus"
    ],
    ans: 1,
    exp: "Documentos e consulta são representados por embeddings (estilo sentence-transformer); um banco de dados vetorial faz k-NN (aproximado) por similaridade de cosseno; as principais passagens são injetadas no prompt. Componentes clássicos, aplicação de fronteira."
  },
  {
    q: "Para prever churn a partir de uma tabela estruturada de clientes, as evidências ainda favorecem...",
    opts: [
      "dar um prompt a um grande modelo de linguagem com cada linha",
      "gradient boosting sobre atributos bem construídos, validado honestamente",
      "classificação zero-shot",
      "treinar um transformer gigante do zero sobre a tabela"
    ],
    ans: 1,
    exp: "LLMs são fracos em predição tabular numérica, caros em escala e difíceis de auditar. As árvores com boosting continuam líderes de benchmark ali — saber quando NÃO usar um modelo de fundação é uma habilidade de fronteira."
  },
  {
    q: "A contaminação de benchmarks — questões de teste aparecendo nos dados de treino de um modelo — é a versão em escala de fronteira de qual conceito deste curso?",
    opts: [
      "o trade-off viés-variância",
      "o vazamento de dados: avaliar em dados que o modelo efetivamente já viu infla os escores",
      "a regressão à média",
      "a maldição da dimensionalidade"
    ],
    ans: 1,
    exp: "É exatamente a contaminação treino/teste da aula de validação, em escala de internet: o modelo 'memorizou a prova'. A avaliação honesta — nova, separada, não contaminada — continua sendo a disciplina mais difícil do ML."
  },
  {
    q: "Qual situação genuinamente pede um modelo de fundação em vez da pilha clássica?",
    opts: [
      "escorar 50 milhões de linhas de transações em um batch noturno dentro do orçamento",
      "extrair intenção e entidades de mensagens de clientes em texto livre com poucos exemplos rotulados",
      "uma decisão de crédito regulada que precisa de contribuições de atributos auditáveis",
      "um conjunto de 500 linhas de leituras de sensores"
    ],
    ans: 1,
    exp: "Linguagem não estruturada com rótulos escassos é exatamente onde o conhecimento pré-treinado compensa. Os outros três — escala/custo, auditabilidade, dados estruturados minúsculos — favorecem a caixa de ferramentas clássica."
  }
]);
</script>
