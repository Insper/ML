import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "the dog chased the cat",
    "the cat chased the mouse",
    "the mouse ate cheese",
    "machine learning models learn from data",
]

vec = TfidfVectorizer()
X = vec.fit_transform(corpus)

df = pd.DataFrame(
    X.toarray().round(2),
    columns=vec.get_feature_names_out(),
    index=[f"doc{i + 1}" for i in range(len(corpus))],
)

styles = (
    "<style>"
    ".tfidf-table table{border-collapse:collapse;font-size:.75rem;}"
    ".tfidf-table th,.tfidf-table td{border:1px solid #888;padding:2px 8px;text-align:center;}"
    "</style>"
)
print(styles + '<div class="tfidf-table" style="overflow-x:auto;">' + df.to_html() + "</div>")
