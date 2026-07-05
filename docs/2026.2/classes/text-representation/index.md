# Text Representation

Everything we have done so far assumed data arrives as a numeric table. Text does not. This lesson builds the bridge — **from raw strings to vectors** — that every text application depends on, and paves the way for [Topic Modeling & BERTopic](../topic-modeling-bertopic/index.md).

The central question: *how do we turn a document into a vector so that geometric tools (distances, clustering, classifiers) apply?*

## Tokenization and the vocabulary

The first step is splitting text into units — **tokens** (words, subwords, or characters) — and building a **vocabulary**: the set of distinct tokens across the corpus. Common normalizations: lowercase, strip punctuation, optionally remove **stop words** ("the", "of", "and" — frequent words carrying little content) and reduce words to stems or lemmas ("running" → "run").

## Bag-of-Words

The classical representation (a direct descendant of 1950s information retrieval): represent a document by its **word counts**, ignoring order entirely.

\[
\text{doc}_i \;\longrightarrow\; x_i \in \mathbb{R}^{|V|}, \qquad x_{ij} = \text{count of vocabulary word } j \text{ in document } i
\]

```python
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
X = vec.fit_transform(corpus)   # sparse matrix: n_docs × |V|
```

"Bag" is literal: *"the dog bit the man"* and *"the man bit the dog"* get identical vectors. Word order — and thus most syntax — is discarded.

## TF-IDF

Raw counts overweight common words. **TF-IDF** (term frequency × inverse document frequency, Spärck Jones 1972) rescales each count by how *distinctive* the word is across the corpus:

\[
\text{tf-idf}(t, d) = \underbrace{\text{tf}(t, d)}_{\text{freq. of } t \text{ in } d} \times \underbrace{\log \frac{1 + n}{1 + \text{df}(t)} + 1}_{\text{idf: rarity of } t \text{ across docs}}
\]

where \(n\) is the number of documents and \(\text{df}(t)\) is the number of documents containing \(t\). A word appearing in *every* document (idf ≈ 1 after smoothing) is discounted; a word concentrated in few documents is amplified. scikit-learn then normalizes each document vector to unit length.

--8<-- "docs/2026.2/classes/text-representation/tfidf-demo.html"

Note how "the" — present in three of four documents — gets low weights, while distinctive words like "cheese" and "learning" score high in their documents. Documents are compared with **cosine similarity**:

\[
\cos(x_a, x_b) = \frac{x_a \cdot x_b}{\lVert x_a \rVert\, \lVert x_b \rVert}
\]

### n-grams: buying back a little order

Counting sequences of \(n\) consecutive tokens (bigrams: "not good", "machine learning") recovers local word order:

```python
TfidfVectorizer(ngram_range=(1, 2))   # unigrams + bigrams
```

The cost: the vocabulary — and the dimensionality — explodes combinatorially.

## The limits of sparse representations

Bag-of-words/TF-IDF vectors are **sparse** (mostly zeros), **high-dimensional** (\(|V|\) can be 10⁵+), and — critically — treat words as **atomic symbols**:

- "car" and "automobile" are orthogonal dimensions: similarity zero, despite being synonyms;
- "bank" (river) and "bank" (finance) are the same dimension: context is invisible;
- a document about *dogs* and one about *puppies* may share no vocabulary and be judged unrelated.

These are exactly the failures that motivated **dense embeddings**.

## Dense embeddings

### Word embeddings — word2vec intuition

**word2vec** (Mikolov et al., 2013) learns a dense vector (~300 dims) per word by training a shallow network to predict words from their contexts. The **distributional hypothesis** does the magic: *words appearing in similar contexts get similar vectors*. Synonyms land close together, and directions encode relations — the famous

\[
\text{vec}(\text{king}) - \text{vec}(\text{man}) + \text{vec}(\text{woman}) \approx \text{vec}(\text{queen}).
\]

GloVe (2014) and fastText (2016) refine the idea. Limitation: **one vector per word**, so "bank" still has a single meaning averaged over all its uses.

### Sentence embeddings — contextual and whole-document

Transformer models (BERT, 2018) produce **contextual** embeddings — the vector for "bank" differs in "river bank" vs "bank loan". **Sentence-BERT** (Reimers & Gurevych, 2019) fine-tunes such models so that a whole sentence or short document maps to a single dense vector (~384–768 dims) where **cosine similarity ≈ semantic similarity**:

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
emb = model.encode(["The dog chased the cat",
                    "A hound pursued a feline",
                    "Interest rates rose again"])
# emb.shape == (3, 384); emb[0]·emb[1] high, emb[0]·emb[2] low
```

The first two sentences share almost no vocabulary yet get highly similar vectors — precisely what TF-IDF cannot do.

## Sparse vs dense: when to use which

| | Sparse (BoW / TF-IDF) | Dense (embeddings) |
|---|---|---|
| Dimensionality | \(|V|\) (10⁴–10⁶), sparse | 300–1000, dense |
| Synonyms | orthogonal (missed) | nearby vectors |
| Polysemy | conflated | handled (contextual models) |
| Interpretability | high — dimensions are words | low — dimensions are abstract |
| Compute cost | trivial | needs a pretrained model |
| Great for | keyword search, linear baselines, [Naive Bayes](../naive-bayes/index.md) | semantic search, clustering, [BERTopic](../topic-modeling-bertopic/index.md) |

!!! tip "Where this leads"
    Next lesson: embed documents densely, reduce with [UMAP](../dimensionality-reduction/index.md), cluster with [HDBSCAN](../clustering/index.md), and describe each cluster with a TF-IDF variant. Sparse *and* dense representations, working together — that pipeline is **BERTopic**.

---

## Quiz

<div id="quiz-text-representation"></div>
<script>
buildQuiz('text-representation', 'Text Representation', [
  {
    q: "In the bag-of-words model, the sentences 'the dog bit the man' and 'the man bit the dog' receive...",
    opts: [
      "opposite vectors",
      "identical vectors, because word order is discarded",
      "orthogonal vectors",
      "vectors of different dimensionality"
    ],
    ans: 1,
    exp: "BoW keeps only counts per vocabulary word. Both sentences contain the same words with the same counts, so their vectors are identical — syntax and order are lost."
  },
  {
    q: "What does the IDF factor in TF-IDF accomplish?",
    opts: [
      "It boosts the most frequent words in the corpus",
      "It down-weights words that appear in many documents and up-weights distinctive, rarer words",
      "It removes stop words from the vocabulary",
      "It normalizes document length"
    ],
    ans: 1,
    exp: "idf = log((1+n)/(1+df)) + 1 shrinks toward its minimum for words present in nearly all documents ('the') and grows for words concentrated in few — making distinctive words dominate the representation."
  },
  {
    q: "Why do 'car' and 'automobile' have cosine similarity 0 under TF-IDF?",
    opts: [
      "Because they have different lengths",
      "Because each word is its own dimension: synonyms are distinct atomic symbols with no shared components",
      "Because IDF removes both words",
      "Because cosine similarity does not apply to sparse vectors"
    ],
    ans: 1,
    exp: "Sparse representations treat the vocabulary as orthogonal axes. Two documents using disjoint (but synonymous) vocabulary share no nonzero dimensions — the core motivation for dense embeddings."
  },
  {
    q: "The famous king − man + woman ≈ queen result illustrates that word2vec...",
    opts: [
      "memorizes an analogy dictionary",
      "learns a vector space where directions capture semantic relations from co-occurrence patterns",
      "uses one-hot encodings",
      "requires labeled analogy data for training"
    ],
    ans: 1,
    exp: "Word2vec trains on raw text, predicting words from contexts. The distributional hypothesis — similar contexts ⇒ similar vectors — produces geometry where relations like gender or capital-of appear as consistent vector offsets."
  },
  {
    q: "What key limitation of word2vec do contextual models like BERT fix?",
    opts: [
      "Word2vec vectors are too small",
      "Word2vec assigns one fixed vector per word, conflating different senses like 'bank' (river) and 'bank' (finance)",
      "Word2vec only works in English",
      "Word2vec cannot be trained on large corpora"
    ],
    ans: 1,
    exp: "Contextual models compute the representation of each token from its surrounding sentence, so the same word gets different vectors in different contexts — resolving polysemy."
  },
  {
    q: "Using ngram_range=(1,2) in TfidfVectorizer buys back some word order. What is the main cost?",
    opts: [
      "Loss of interpretability",
      "The vocabulary (and dimensionality) grows combinatorially",
      "It removes unigrams from the representation",
      "Cosine similarity stops working"
    ],
    ans: 1,
    exp: "Every adjacent word pair becomes a potential feature, multiplying vocabulary size — more memory, more sparsity, higher risk of overfitting on rare n-grams."
  }
]);
</script>
