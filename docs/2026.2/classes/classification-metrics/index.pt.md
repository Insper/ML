# Classificação & Métricas

A **classificação** é o aprendizado supervisionado com alvo categórico: spam/não spam, inadimplente/pagador, doente/saudável, ou uma dentre muitas classes (dígito 0–9). As Partes IV e V constroem um portfólio de classificadores; esta aula constrói a ferramenta de que você precisa *antes* de qualquer um deles — saber **como medir se um classificador é bom**. Escolher a métrica errada não é um detalhe: silenciosamente otimiza o comportamento errado.

## A matriz de confusão

Para um problema binário, chame uma classe de **positiva** (geralmente a rara/interessante: fraude, doença) e a outra de **negativa**. Toda previsão cai em uma de quatro células:

|  | Previsto positivo | Previsto negativo |
|---|---|---|
| **Realmente positivo** | VP (verdadeiro positivo) | FN (falso negativo) — *escape* |
| **Realmente negativo** | FP (falso positivo) — *alarme falso* | VN (verdadeiro negativo) |

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
confusion_matrix(y_test, y_pred)          # linhas = verdade, colunas = previsão
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
```

Os dois tipos de erro geralmente têm **custos muito diferentes**: um câncer não detectado (FN) não é o mesmo que um exame de acompanhamento desnecessário (FP); uma transação legítima bloqueada (FP) não é o mesmo que uma fraude aprovada (FN). As métricas existem para pesá-los explicitamente.

## Acurácia — e por que ela mente

\[
\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN}
\]

A acurácia responde "que fração das previsões estava certa?" — razoável quando as classes são balanceadas e os erros custam o mesmo. Mas com **classes desbalanceadas** ela degenera. Fraude é 0,5% das transações? A regra burra "tudo é legítimo" pontua **99,5% de acurácia** enquanto pega *zero* fraude.

!!! danger "O paradoxo da acurácia"
    Em problemas desbalanceados, alta acurácia pode descrever um modelo inútil. Sempre compare com a baseline da classe majoritária (`DummyClassifier`) e recorra às métricas abaixo.

## Precisão e revocação

Ambas focam na classe positiva, respondendo a perguntas diferentes:

\[
\text{Precisão} = \frac{VP}{VP + FP}
\qquad\qquad
\text{Revocação} = \frac{VP}{VP + FN}
\]

- **Precisão** — *dos casos que sinalizei, quantos eram reais?* Precisão alta = poucos alarmes falsos;
- **Revocação** (sensibilidade) — *dos casos reais, quantos eu peguei?* Revocação alta = poucos escapes.

Elas puxam em direções opostas: sinalize mais agressivamente e a revocação sobe enquanto a precisão cai; sinalize com cautela e o inverso. Qual priorizar é uma **decisão de domínio**:

| Aplicação | Erro custoso | Priorizar |
|-----------|--------------|-----------|
| Rastreio de câncer | deixar de detectar um paciente (FN) | revocação |
| Filtro de spam | perder um e-mail real (FP) | precisão |
| Triagem de fraude para revisão humana | desperdiçar tempo do analista (FP) vs fraude não detectada (FN) | equilíbrio — depende da capacidade |

### F1: um número quando você precisa de um só

A **média harmônica** de precisão e revocação:

\[
F_1 = 2 \cdot \frac{\text{Precisão} \cdot \text{Revocação}}{\text{Precisão} + \text{Revocação}}
\]

A média harmônica pune o desequilíbrio: precisão 1,0 com revocação 0,02 dá \(F_1 \approx 0.04\), não a média aritmética 0,51 — você não consegue comprar um bom F1 maximizando um lado só. O \(F_\beta\) geral pesa a revocação \(\beta\) vezes mais que a precisão (\(F_2\) para rastreio, \(F_{0.5}\) para spam).

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

## Multiclasse

A matriz de confusão se generaliza para \(k \times k\) — as células fora da diagonal revelam *quais* classes são confundidas (diagnósticos úteis: o modelo está misturando 4 e 9?). Precisão/revocação/F1 por classe são combinados por:

- média **macro** — média sobre as classes, todas iguais (classes pequenas contam integralmente);
- média **ponderada** (weighted) — média ponderada pela frequência da classe;
- média **micro** — calculada a partir das contagens agregadas (igual à acurácia em problemas de rótulo único).

Em dados multiclasse desbalanceados, reporte o **macro-F1**: ele expõe falhas em classes raras que as médias ponderadas escondem.

## Escores, limiares e calibração

A maioria dos classificadores produz um **escore ou probabilidade**, e o rótulo vem de um **limiar** (padrão 0,5):

```python
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= 0.5).astype(int)       # o limiar é uma escolha!
```

Mover o limiar troca precisão por revocação — abaixe-o para pegar mais positivos (revocação ↑, precisão ↓), aumente-o para alarmes mais limpos. O limiar é uma **decisão de negócio aplicada após o treino**, e avaliar um modelo *em todos os limiares* é exatamente o que as curvas ROC e precisão–revocação fazem — o tema de [ROC-AUC & Dados Desbalanceados](../roc-imbalanced/index.md).

Quando as próprias probabilidades previstas importam (precificação de risco, ordenação de triagem), verifique a **calibração**: entre os casos previstos como "70%", cerca de 70% acabam sendo positivos? (`sklearn.calibration.CalibrationDisplay`).

## Material de aula

!!! example "Notebook da aula (em português)"
    Notebook prático usado em sala — **Aula 13 — Classificação de Dados**:
    [:simple-googlecolab: abrir no Colab](https://colab.research.google.com/drive/1Kw9tOXyDIrS3r9dsZr7K9drfT9zBdyGx){:target="_blank"}

---

## Quiz

<div id="quiz-classification-metrics"></div>
<script>
buildQuiz('classification-metrics', 'Classificação & Métricas', [
  {
    q: "Uma doença afeta 1% dos pacientes. Um modelo que diz 'saudável' para todos alcança...",
    opts: [
      "99% de acurácia e 0% de revocação para a classe da doença",
      "99% de acurácia e 99% de revocação",
      "50% de acurácia",
      "0% de acurácia"
    ],
    ans: 0,
    exp: "Ele acerta os 99% de pacientes saudáveis (acurácia 0,99) mas não pega nenhum dos doentes (VP = 0 → revocação 0). O paradoxo da acurácia: em dados desbalanceados, a acurácia pode lisonjear um modelo inútil."
  },
  {
    q: "A precisão responde à pergunta...",
    opts: [
      "De todos os positivos reais, quantos o modelo pegou?",
      "De todos os casos que o modelo sinalizou como positivos, quantos realmente eram?",
      "Quantas previsões estavam corretas no total?",
      "Quão bem calibradas estão as probabilidades?"
    ],
    ans: 1,
    exp: "Precisão = VP/(VP+FP): a pureza dos alarmes do modelo. Revocação = VP/(VP+FN) responde à primeira opção — a cobertura dos positivos reais."
  },
  {
    q: "Para um teste de rastreio de câncer, qual erro costuma ser considerado pior e, portanto, qual métrica deve ser priorizada?",
    opts: [
      "Falsos positivos são piores — priorize a precisão",
      "Falsos negativos são piores (um paciente não detectado) — priorize a revocação",
      "Ambos são igualmente ruins — use acurácia",
      "Nenhum — o rastreio não usa classificação"
    ],
    ans: 1,
    exp: "Um câncer não detectado pode ser fatal; um alarme falso leva a um exame de acompanhamento. O rastreio favorece a revocação (sensibilidade), aceitando mais falsos positivos — que os testes confirmatórios seguintes então filtram."
  },
  {
    q: "Um modelo tem precisão 1,00 e revocação 0,02. Seu F1 é aproximadamente...",
    opts: [
      "0,51, a média dos dois",
      "0,04 — a média harmônica colapsa quando um componente é minúsculo",
      "1,00, porque a precisão é perfeita",
      "0,98"
    ],
    ans: 1,
    exp: "F1 = 2PR/(P+R) = 2·1·0,02/1,02 ≈ 0,039. A média harmônica é dominada pelo componente mais fraco, então você não consegue burlar o F1 maximizando apenas um lado."
  },
  {
    q: "Baixar o limiar de decisão de 0,5 para 0,2 costuma causar...",
    opts: [
      "aumento da revocação e queda da precisão",
      "aumento da precisão e queda da revocação",
      "aumento de ambas",
      "nenhuma mudança — o limiar só afeta o treino"
    ],
    ans: 0,
    exp: "Uma barra mais baixa sinaliza mais casos como positivos: mais verdadeiros positivos são pegos (revocação ↑) mas mais negativos também são arrastados (precisão ↓). O limiar é uma decisão de negócio pós-treino."
  },
  {
    q: "Em um problema desbalanceado de 10 classes, por que reportar o macro-F1 em vez de acurácia ou F1 ponderado?",
    opts: [
      "O macro-F1 é sempre o maior número",
      "O macro-F1 faz a média dos F1 por classe com peso igual, então o colapso em classes raras fica visível em vez de ser mascarado pelas classes frequentes",
      "O F1 ponderado não pode ser calculado para 10 classes",
      "A acurácia é indefinida para problemas multiclasse"
    ],
    ans: 1,
    exp: "A acurácia e as médias ponderadas são dominadas pelas classes grandes; um modelo pode falhar em toda classe rara e ainda parecer bom. O macro trata cada classe como igualmente importante, expondo essas falhas."
  }
]);
</script>
