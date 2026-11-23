"""The two students, written with numpy so the math stays visible.

1. TF-IDF features plus softmax (multinomial logistic) regression.
2. Sentence embeddings plus the same regression. FastEmbed gives the embeddings when its model
   can download. Otherwise a hash embedder stands in: hashed character n-grams, which is not a
   learned embedding. Tests always use the hash embedder, so they run offline.
"""

from __future__ import annotations

import math
import re
import zlib
from collections import Counter

import numpy as np

TOKEN = re.compile(r"[a-z0-9']+")


def tokens(text: str) -> list[str]:
    words = TOKEN.findall(text.lower())
    return words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]


class Tfidf:
    """Word unigrams and bigrams, smooth IDF, sublinear TF and L2 rows. Keeps the top features by DF."""

    def __init__(self, max_features: int = 6000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf = np.zeros(0, dtype=np.float32)

    def fit(self, texts: list[str]) -> Tfidf:
        df = Counter()
        for t in texts:
            df.update(set(tokens(t)))
        kept = [w for w, c in df.items() if c >= self.min_df] or list(df)
        kept.sort(key=lambda w: (-df[w], w))
        kept = sorted(kept[: self.max_features])
        self.vocab = {w: i for i, w in enumerate(kept)}
        n = len(texts)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in kept], dtype=np.float32)
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for r, t in enumerate(texts):
            for w, c in Counter(tokens(t)).items():
                j = self.vocab.get(w)
                if j is not None:
                    x[r, j] = 1 + math.log(c)
        x *= self.idf
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)


class HashEmbedder:
    """Character 3 to 5 grams hashed into `dim` buckets with a sign bit, then L2 rows.

    Deterministic across runs and machines because it uses crc32, not Python's salted hash().
    """

    name = "hash"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def fit(self, texts: list[str]) -> HashEmbedder:
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        x = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, t in enumerate(texts):
            s = f" {' '.join(TOKEN.findall(t.lower()))} "
            for n in (3, 4, 5):
                for i in range(len(s) - n + 1):
                    h = zlib.crc32(s[i : i + n].encode())
                    x[r, h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norms, 1e-12)


class FastEmbedder:
    """FastEmbed sentence embeddings. Needs the optional `fastembed` extra and a model download."""

    name = "fastembed"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding  # imported here so the core needs only numpy

        self.model_name = model
        self.model = TextEmbedding(model)

    def fit(self, texts: list[str]) -> FastEmbedder:
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.asarray(list(self.model.embed(texts)), dtype=np.float32)


class SoftmaxRegression:
    """Multinomial logistic regression trained with full-batch Adam and L2 weight decay."""

    def __init__(self, l2: float = 1e-4, lr: float = 0.1, epochs: int = 300, seed: int = 0):
        self.l2, self.lr, self.epochs, self.seed = l2, lr, epochs, seed
        self.classes: list[str] = []

    def fit(self, x: np.ndarray, labels: list[str]) -> SoftmaxRegression:
        self.classes = sorted(set(labels))
        index = {c: i for i, c in enumerate(self.classes)}
        y = np.array([index[v] for v in labels])
        n, d = x.shape
        k = len(self.classes)
        onehot = np.zeros((n, k), dtype=np.float32)
        onehot[np.arange(n), y] = 1
        rng = np.random.default_rng(self.seed)
        self.w = (rng.standard_normal((d, k)) * 0.01).astype(np.float32)
        self.b = np.zeros(k, dtype=np.float32)
        mw, vw = np.zeros_like(self.w), np.zeros_like(self.w)
        mb, vb = np.zeros_like(self.b), np.zeros_like(self.b)
        b1, b2, eps = 0.9, 0.999, 1e-8
        for t in range(1, self.epochs + 1):
            p = self._softmax(x @ self.w + self.b)
            g = (p - onehot) / n
            gw = x.T @ g + self.l2 * self.w
            gb = g.sum(axis=0)
            mw = b1 * mw + (1 - b1) * gw
            vw = b2 * vw + (1 - b2) * gw * gw
            mb = b1 * mb + (1 - b1) * gb
            vb = b2 * vb + (1 - b2) * gb * gb
            c1, c2 = 1 - b1**t, 1 - b2**t
            self.w -= self.lr * (mw / c1) / (np.sqrt(vw / c2) + eps)
            self.b -= self.lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        return self

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._softmax(x @ self.w + self.b)


class Student:
    """Features plus regression. `predict` returns labels and the top probability as confidence."""

    def __init__(self, kind: str = "tfidf", seed: int = 0, embedder=None):
        self.kind = kind
        if kind == "tfidf":
            self.features = Tfidf()
        elif kind == "embed":
            self.features = embedder or HashEmbedder()
        else:
            raise ValueError(f"unknown student {kind!r}")
        self.model = SoftmaxRegression(seed=seed)

    @property
    def name(self) -> str:
        return "tfidf" if self.kind == "tfidf" else f"embed-{self.features.name}"

    def fit(self, texts: list[str], labels: list[str]) -> Student:
        x = self.features.fit(texts).transform(texts)
        self.model.fit(x, labels)
        return self

    def predict(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        p = self.model.predict_proba(self.features.transform(texts))
        top = p.argmax(axis=1)
        return [self.model.classes[i] for i in top], p.max(axis=1)
