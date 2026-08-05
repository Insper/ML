# Naive Bayes

O Naive Bayes é o ML clássico em sua forma mais pura: aplique um teorema do século XVIII (Bayes, 1763) com uma ousada suposição simplificadora e obtenha um classificador que é rápido, precisa de poucos dados e filtrou seu spam por duas décadas. Ele também religa a Parte IV à [representação de texto](../text-representation/index.md) — o bag-of-words é seu habitat natural.

## O teorema de Bayes

Para uma classe \(y\) e atributos observados \(x\):

\[
\underbrace{P(y \mid x)}_{\text{posteriori}} = \frac{\overbrace{P(x \mid y)}^{\text{verossimilhança}}\;\overbrace{P(y)}^{\text{priori}}}{\underbrace{P(x)}_{\text{evidência}}}
\]

Leia como uma regra de aprendizado: parta da **priori** (quão comum é cada classe?), pondere pela **verossimilhança** (quão típicos são esses atributos para essa classe?) e obtenha a **posteriori** (quão provável é a classe, dado o que observamos?). Classifique pela maior posteriori — \(P(x)\) é a mesma para todas as classes e se cancela:

\[
\hat{y} = \arg\max_y \; P(y)\, P(x \mid y)
\]

## A suposição "ingênua"

A verossimilhança \(P(x_1, \dots, x_d \mid y)\) é uma distribuição conjunta sobre todas as combinações de atributos — impossível de estimar. O Naive Bayes assume que os atributos são **condicionalmente independentes dada a classe**:

\[
P(x_1, \dots, x_d \mid y) \;\approx\; \prod_{j=1}^{d} P(x_j \mid y)
\qquad\Longrightarrow\qquad
\hat{y} = \arg\max_y \; P(y) \prod_{j=1}^{d} P(x_j \mid y)
\]

Agora cada \(P(x_j \mid y)\) é uma estimativa unidimensional simples: conte frequências (atributos discretos) ou ajuste uma gaussiana (contínuos). Treinar = contar. Uma passada pelos dados.

A suposição é quase sempre **falsa** (em spam real, "grátis" e "oferta" coocorrem muito mais do que a independência prevê). Por que ainda funciona? A classificação precisa apenas da **ordenação** das posterioris, não de seus valores exatos. O Naive Bayes costuma acertar o \(\arg\max\) mesmo quando as próprias probabilidades estão muito distorcidas — tipicamente superconfiantes (empurradas para 0 ou 1). Confie em seus rótulos, não em suas probabilidades.

## O clássico: filtragem de spam

Com atributos de [bag-of-words](../text-representation/index.md#bag-of-words), o **Multinomial Naive Bayes** modela contagens de palavras por classe. Estime a partir das contagens de treino:

\[
P(\text{spam}) = \frac{\#\text{docs spam}}{\#\text{docs}},
\qquad
P(w \mid \text{spam}) = \frac{\text{count}(w, \text{spam}) + 1}{\sum_{w'} \text{count}(w', \text{spam}) + |V|}
\]

O \(+1\) é a **suavização de Laplace**: sem ela, uma única palavra nunca vista nos dados de treino de spam dá \(P(w \mid \text{spam}) = 0\), e um zero **aniquila o produto inteiro** — qualquer e-mail contendo essa palavra jamais poderia ser spam. A suavização finge que cada palavra foi vista uma vez.

Na prática, multiplicar muitos números pequenos causa underflow, então as implementações somam logs:

\[
\hat{y} = \arg\max_y \Big[ \log P(y) + \sum_j \log P(x_j \mid y) \Big]
\]

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

spam_filter = make_pipeline(CountVectorizer(), MultinomialNB(alpha=1.0))  # alpha = suavização
spam_filter.fit(docs_train, y_train)
spam_filter.predict(["WIN a FREE prize now!!!"])
```

## A família Naive Bayes

| Variante | Tipo de atributo | Modelo de \(P(x_j \mid y)\) | Uso típico |
|----------|------------------|-----------------------------|------------|
| `MultinomialNB` | contagens | multinomial sobre contagens | texto (bag-of-words / TF-IDF) |
| `BernoulliNB` | binário | Bernoulli (presença/ausência) | texto curto, flags binárias |
| `GaussianNB` | contínuo | gaussiana por atributo por classe | dados tabulares numéricos |

Para o `GaussianNB`, cada classe simplesmente armazena uma média e uma variância por atributo:

\[
P(x_j \mid y) = \frac{1}{\sqrt{2\pi\sigma_{jy}^2}} \exp\!\Big(-\frac{(x_j - \mu_{jy})^2}{2\sigma_{jy}^2}\Big)
\]

## Perfil prático

| | |
|---|---|
| **Pontos fortes** | treina em uma passada (o aprendiz mais rápido do curso); funciona com poucos dados; lida com dimensões muito altas (10⁵ atributos de palavras); naturalmente multiclasse; atualizável online (`partial_fit`); não precisa de escalonamento |
| **Fraquezas** | estimativas de probabilidade mal calibradas; atributos correlacionados contam a evidência em dobro; fronteiras de decisão quase lineares; a ficção da independência pode realmente prejudicar |
| **Recorra a ele quando** | baselines de texto, muitos atributos + poucas amostras, previsão crítica em latência ou em streaming |

!!! tip "A baseline perfeita"
    Qualquer que seja o modelo sofisticado que você planeja para um problema de texto, ajuste `CountVectorizer + MultinomialNB` primeiro. Leva segundos, e qualquer coisa que não consiga superá-lo não vale a pena implantar.

---

## Quiz

<div id="quiz-naive-bayes"></div>
<script>
buildQuiz('naive-bayes', 'Naive Bayes', [
  {
    q: "No teorema de Bayes aplicado à classificação, a priori P(y) representa...",
    opts: [
      "a probabilidade dos atributos",
      "quão comum é cada classe antes de observar qualquer atributo",
      "a confiança do modelo depois de ver os dados",
      "o parâmetro de suavização"
    ],
    ans: 1,
    exp: "A priori codifica as taxas base (ex.: 40% dos e-mails são spam). A verossimilhança P(x|y) então a atualiza com a evidência dos atributos, gerando a posteriori P(y|x)."
  },
  {
    q: "A suposição 'ingênua' afirma que...",
    opts: [
      "todas as classes são igualmente prováveis",
      "os atributos são condicionalmente independentes dada a classe, então a verossimilhança conjunta se fatoriza em um produto de termos por atributo",
      "os atributos seguem uma distribuição uniforme",
      "os dados de treino não têm ruído"
    ],
    ans: 1,
    exp: "P(x₁,...,x_d|y) ≈ ∏P(x_j|y). Isso reduz um problema de estimação conjunta impossível a d problemas unidimensionais — contáveis a partir dos dados em uma única passada."
  },
  {
    q: "Por que o Naive Bayes costuma classificar bem apesar de sua suposição de independência ser falsa?",
    opts: [
      "A suposição é de fato verdadeira para a maioria dos conjuntos de dados",
      "A classificação só precisa do argmax entre as classes: a ordenação das posterioris muitas vezes sobrevive mesmo a estimativas de probabilidade muito distorcidas",
      "O scikit-learn corrige a suposição internamente",
      "Não classifica — tem sempre desempenho ruim"
    ],
    ans: 1,
    exp: "As violações distorcem as magnitudes da posteriori (tipicamente para superconfiança) mas frequentemente preservam qual classe pontua mais alto. Daí: confie nos rótulos, desconfie das probabilidades."
  },
  {
    q: "Sem a suavização de Laplace, uma palavra que nunca apareceu nos dados de treino de spam faria...",
    opts: [
      "ser ignorada pelo modelo",
      "P(spam | qualquer e-mail que a contenha) = 0, já que um único fator zero aniquila o produto inteiro",
      "aumentar a priori de spam",
      "causar uma divisão por zero na priori"
    ],
    ans: 1,
    exp: "A verossimilhança é um produto; um único P(w|spam) = 0 o zera independentemente de toda a demais evidência. Adicionar 1 a cada contagem (α = 1) garante que nenhuma probabilidade seja exatamente zero."
  },
  {
    q: "Para classificar e-mails representados como contagens de palavras, a variante apropriada é...",
    opts: [
      "GaussianNB",
      "MultinomialNB",
      "BernoulliNB com atributos contínuos",
      "LinearRegression"
    ],
    ans: 1,
    exp: "O MultinomialNB modela dados de contagem — o par natural do bag-of-words. O GaussianNB é para atributos contínuos; o BernoulliNB para vetores binários de presença/ausência."
  },
  {
    q: "Dois atributos estão quase duplicados no conjunto de dados. Para o Naive Bayes isso causa...",
    opts: [
      "nenhum efeito — os atributos são independentes",
      "a evidência compartilhada a ser contada duas vezes, empurrando as posterioris para a superconfiança",
      "um erro de treino",
      "a remoção automática de um atributo"
    ],
    ans: 1,
    exp: "Independência-dada-a-classe significa que todo atributo multiplica sua evidência. Atributos duplicados (correlacionados) injetam a mesma evidência várias vezes — um jeito concreto de a suposição ingênua morder."
  }
]);
</script>
