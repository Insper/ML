# Representação de Texto

Tudo o que fizemos até agora supôs que os dados chegam como uma tabela numérica. Texto não chega. Esta aula constrói a ponte — **de strings brutas a vetores** — de que toda aplicação de texto depende, e prepara o terreno para [Modelagem de Tópicos & BERTopic](../topic-modeling-bertopic/index.md).

A pergunta central: *como transformar um documento em um vetor para que ferramentas geométricas (distâncias, agrupamento, classificadores) se apliquem?*

## Tokenização e o vocabulário

O primeiro passo é dividir o texto em unidades — **tokens** (palavras, subpalavras ou caracteres) — e construir um **vocabulário**: o conjunto de tokens distintos em todo o corpus. Normalizações comuns: minúsculas, remover pontuação, opcionalmente remover **stop words** ("o", "de", "e" — palavras frequentes que carregam pouco conteúdo) e reduzir palavras a radicais (stems) ou lemas ("correndo" → "correr").

## Bag-of-Words

A representação clássica (descendente direta da recuperação de informação dos anos 1950): representar um documento por suas **contagens de palavras**, ignorando totalmente a ordem.

\[
\text{doc}_i \;\longrightarrow\; x_i \in \mathbb{R}^{|V|}, \qquad x_{ij} = \text{contagem da palavra } j \text{ do vocabulário no documento } i
\]

```python
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(corpus)   # matriz esparsa: n_docs × |V|
```

"Bag" (saco) é literal: *"o cachorro mordeu o homem"* e *"o homem mordeu o cachorro"* recebem vetores idênticos. A ordem das palavras — e, portanto, a maior parte da sintaxe — é descartada.

## TF-IDF

Contagens brutas dão peso demais a palavras comuns. O **TF-IDF** (frequência do termo × frequência inversa nos documentos, Spärck Jones 1972) reescala cada contagem por quão *distintiva* a palavra é no corpus:

\[
\text{tf-idf}(t, d) = \underbrace{\text{tf}(t, d)}_{\text{freq. de } t \text{ em } d} \times \underbrace{\log \frac{1 + n}{1 + \text{df}(t)} + 1}_{\text{idf: raridade de } t \text{ nos docs}}
\]

onde \(n\) é o número de documentos e \(\text{df}(t)\) é o número de documentos que contêm \(t\). Uma palavra que aparece em *todos* os documentos (idf ≈ 1 após a suavização) é descontada; uma palavra concentrada em poucos documentos é amplificada. O scikit-learn então normaliza cada vetor de documento para comprimento unitário.

--8<-- "docs/2026.2/classes/text-representation/tfidf-demo.html"

Note como "the" — presente em três de quatro documentos — recebe pesos baixos, enquanto palavras distintivas como "cheese" e "learning" pontuam alto em seus documentos. Documentos são comparados com **similaridade de cosseno**:

\[
\cos(x_a, x_b) = \frac{x_a \cdot x_b}{\lVert x_a \rVert\, \lVert x_b \rVert}
\]

### n-gramas: comprando de volta um pouco de ordem

Contar sequências de \(n\) tokens consecutivos (bigramas: "not good", "machine learning") recupera a ordem local das palavras:

```python
TfidfVectorizer(ngram_range=(1, 2))   # unigramas + bigramas
```

O custo: o vocabulário — e a dimensionalidade — explode combinatoriamente.

## Os limites das representações esparsas

Vetores de bag-of-words/TF-IDF são **esparsos** (majoritariamente zeros), **de alta dimensão** (\(|V|\) pode passar de 10⁵) e — crucialmente — tratam as palavras como **símbolos atômicos**:

- "carro" e "automóvel" são dimensões ortogonais: similaridade zero, apesar de sinônimos;
- "banco" (rio) e "banco" (finanças) são a mesma dimensão: o contexto é invisível;
- um documento sobre *cachorros* e outro sobre *filhotes* podem não compartilhar vocabulário e serem julgados não relacionados.

Essas são exatamente as falhas que motivaram os **embeddings densos**.

## Embeddings densos

### Word embeddings — a intuição do word2vec

O **word2vec** (Mikolov et al., 2013) aprende um vetor denso (~300 dimensões) por palavra treinando uma rede rasa para prever palavras a partir de seus contextos. A **hipótese distribucional** faz a mágica: *palavras que aparecem em contextos semelhantes recebem vetores semelhantes*. Sinônimos ficam próximos, e as direções codificam relações — o famoso

\[
\text{vec}(\text{rei}) - \text{vec}(\text{homem}) + \text{vec}(\text{mulher}) \approx \text{vec}(\text{rainha}).
\]

GloVe (2014) e fastText (2016) refinam a ideia. Limitação: **um vetor por palavra**, então "banco" ainda tem um único sentido, uma média sobre todos os seus usos.

### Sentence embeddings — contextuais e de documento inteiro

Modelos transformer (BERT, 2018) produzem embeddings **contextuais** — o vetor de "banco" difere em "banco do rio" vs "empréstimo do banco". O **Sentence-BERT** (Reimers & Gurevych, 2019) faz o fine-tuning desses modelos para que uma frase inteira ou um documento curto mapeie para um único vetor denso (~384–768 dimensões) onde **similaridade de cosseno ≈ similaridade semântica**:

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(["The dog chased the cat",
                    "A hound pursued a feline",
                    "Interest rates rose again"])
# emb.shape == (3, 384); emb[0]·emb[1] alto, emb[0]·emb[2] baixo
```

As duas primeiras frases quase não compartilham vocabulário, mas recebem vetores altamente similares — exatamente o que o TF-IDF não consegue fazer.

## Esparso vs denso: quando usar cada um

| | Esparso (BoW / TF-IDF) | Denso (embeddings) |
|---|---|---|
| Dimensionalidade | \(|V|\) (10⁴–10⁶), esparso | 300–1000, denso |
| Sinônimos | ortogonais (perdidos) | vetores próximos |
| Polissemia | confundida | tratada (modelos contextuais) |
| Interpretabilidade | alta — as dimensões são palavras | baixa — as dimensões são abstratas |
| Custo computacional | trivial | precisa de um modelo pré-treinado |
| Ótimo para | busca por palavra-chave, baselines lineares | busca semântica, agrupamento, [BERTopic](../topic-modeling-bertopic/index.md) |

!!! tip "Para onde isso leva"
    Próxima aula: representar documentos densamente, reduzir com [UMAP](../dimensionality-reduction/index.md), agrupar com [HDBSCAN](../clustering/index.md) e descrever cada agrupamento com uma variante de TF-IDF. Representações esparsas *e* densas trabalhando juntas — esse pipeline é o **BERTopic**.

---

## Quiz

<div id="quiz-text-representation"></div>
<script>
buildQuiz('text-representation', 'Representação de Texto', [
  {
    q: "No modelo bag-of-words, as frases 'o cachorro mordeu o homem' e 'o homem mordeu o cachorro' recebem...",
    opts: [
      "vetores opostos",
      "vetores idênticos, porque a ordem das palavras é descartada",
      "vetores ortogonais",
      "vetores de dimensionalidade diferente"
    ],
    ans: 1,
    exp: "O BoW mantém apenas as contagens por palavra do vocabulário. Ambas as frases contêm as mesmas palavras com as mesmas contagens, então seus vetores são idênticos — sintaxe e ordem são perdidas."
  },
  {
    q: "O que o fator IDF do TF-IDF realiza?",
    opts: [
      "Aumenta o peso das palavras mais frequentes do corpus",
      "Reduz o peso de palavras que aparecem em muitos documentos e aumenta o de palavras distintivas e mais raras",
      "Remove stop words do vocabulário",
      "Normaliza o comprimento do documento"
    ],
    ans: 1,
    exp: "idf = log((1+n)/(1+df)) + 1 encolhe para o mínimo em palavras presentes em quase todos os documentos ('the') e cresce para palavras concentradas em poucos — fazendo as palavras distintivas dominarem a representação."
  },
  {
    q: "Por que 'carro' e 'automóvel' têm similaridade de cosseno 0 sob TF-IDF?",
    opts: [
      "Porque têm comprimentos diferentes",
      "Porque cada palavra é sua própria dimensão: sinônimos são símbolos atômicos distintos sem componentes em comum",
      "Porque o IDF remove ambas as palavras",
      "Porque a similaridade de cosseno não se aplica a vetores esparsos"
    ],
    ans: 1,
    exp: "Representações esparsas tratam o vocabulário como eixos ortogonais. Dois documentos que usam vocabulário disjunto (mas sinônimo) não compartilham dimensões não nulas — a motivação central para os embeddings densos."
  },
  {
    q: "O famoso resultado rei − homem + mulher ≈ rainha ilustra que o word2vec...",
    opts: [
      "memoriza um dicionário de analogias",
      "aprende um espaço vetorial onde as direções capturam relações semânticas a partir de padrões de coocorrência",
      "usa codificações one-hot",
      "requer dados rotulados de analogias para treinar"
    ],
    ans: 1,
    exp: "O word2vec treina em texto bruto, prevendo palavras a partir de contextos. A hipótese distribucional — contextos semelhantes ⇒ vetores semelhantes — produz uma geometria em que relações como gênero ou capital-de aparecem como deslocamentos vetoriais consistentes."
  },
  {
    q: "Que limitação central do word2vec os modelos contextuais como o BERT corrigem?",
    opts: [
      "Os vetores do word2vec são pequenos demais",
      "O word2vec atribui um vetor fixo por palavra, confundindo sentidos diferentes como 'banco' (rio) e 'banco' (finanças)",
      "O word2vec só funciona em inglês",
      "O word2vec não pode ser treinado em corpora grandes"
    ],
    ans: 1,
    exp: "Modelos contextuais computam a representação de cada token a partir da frase que o cerca, então a mesma palavra recebe vetores diferentes em contextos diferentes — resolvendo a polissemia."
  },
  {
    q: "Usar ngram_range=(1,2) no TfidfVectorizer compra de volta um pouco de ordem das palavras. Qual é o principal custo?",
    opts: [
      "Perda de interpretabilidade",
      "O vocabulário (e a dimensionalidade) cresce combinatoriamente",
      "Remove os unigramas da representação",
      "A similaridade de cosseno para de funcionar"
    ],
    ans: 1,
    exp: "Cada par de palavras adjacentes vira um atributo potencial, multiplicando o tamanho do vocabulário — mais memória, mais esparsidade, maior risco de sobreajuste em n-gramas raros."
  }
]);
</script>
