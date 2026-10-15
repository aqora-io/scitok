"""Retrieval systems under evaluation, behind one interface.

Every system encodes a corpus once (cached under eval/cache) and returns a score
matrix for a list of queries. Sparse systems are scored the way Qdrant scores
them: dot product, optionally with Qdrant's IDF modifier
idf(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
EVAL_DOMAIN = ROOT / "domains" / "quant-ph"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


class Sparse:
    """Sparse vectors from `encode_docs` / `encode_queries` as (indices, values) pairs."""

    idf = True
    name = "sparse"

    def encode_docs(self, texts):
        raise NotImplementedError

    def encode_queries(self, texts):
        raise NotImplementedError

    def fit(self, corpus_name, texts):
        path = CACHE / self.name / f"{corpus_name}.npz"
        if path.exists():
            self.docs = sp.load_npz(path)
        else:
            self.docs = _to_csr(self.encode_docs(texts))
            path.parent.mkdir(parents=True, exist_ok=True)
            sp.save_npz(path, self.docs)
        dims = self.docs.shape[1]
        if self.idf:
            n = len(texts)
            df = np.bincount(self.docs.indices, minlength=dims)
            self.weights = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        else:
            self.weights = np.ones(dims, dtype=np.float32)
        return self

    def scores(self, queries):
        q = _to_csr(self.encode_queries(queries), dims=self.docs.shape[1])
        q = q @ sp.diags(self.weights)
        return (q @ self.docs.T).toarray()


def _to_csr(vectors, dims=None):
    """(indices, values) pairs → CSR. Term ids are u32 hashes (or vocabulary
    ids), folded into 2^24 columns; the folding is the same for documents and
    queries, and extra collisions are negligible at this vocabulary size."""
    rows, cols, vals = [], [], []
    for r, (idx, val) in enumerate(vectors):
        idx = np.asarray(idx, dtype=np.int64) % (1 << 24)
        rows.append(np.full(len(idx), r))
        cols.append(idx)
        vals.append(np.asarray(val, dtype=np.float32))
    rows = np.concatenate(rows) if rows else np.zeros(0)
    cols = np.concatenate(cols) if cols else np.zeros(0)
    vals = np.concatenate(vals) if vals else np.zeros(0)
    m = sp.csr_matrix((vals, (rows, cols)), shape=(len(vectors), dims or (1 << 24)), dtype=np.float32)
    m.sum_duplicates()
    return m


class Scitok(Sparse):
    def __init__(self, name="scitok", domain_paths=(str(EVAL_DOMAIN),), **options):
        import scitok

        self.name = name
        self.kwargs = dict(domains=[], domain_paths=list(domain_paths), **options)
        self.tok = scitok.Tokenizer(**self.kwargs)
        self.name = f"{name}-{_key(self.kwargs)}"

    def encode_docs(self, texts):
        return self.tok.embed_documents(list(texts))

    def encode_queries(self, texts):
        return [self.tok.embed_query(t) for t in texts]

    def fit(self, corpus_name, texts):
        # Never cached: encoding takes seconds, and the binary changes during tuning.
        self.docs = _to_csr(self.encode_docs(texts))
        n = len(texts)
        df = np.bincount(self.docs.indices, minlength=self.docs.shape[1])
        self.weights = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        return self


class Whitespace(Sparse):
    """Lowercased alphanumeric tokens, no stemming or stopwords, BM25 TF saturation."""

    def __init__(self, k1=1.2, b=0.75):
        self.name, self.k1, self.b = "whitespace-bm25", k1, b

    @staticmethod
    def tokens(text):
        return re.findall(r"\w+", text.lower())

    def encode_docs(self, texts):
        toks = [self.tokens(t) for t in texts]
        avg = sum(map(len, toks)) / len(toks)
        out = []
        for ts in toks:
            ids, tf = np.unique([_hash(t) for t in ts], return_counts=True)
            sat = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * len(ts) / avg))
            out.append((ids, sat))
        return out

    def encode_queries(self, texts):
        ids = [np.unique([_hash(t) for t in self.tokens(q)]) for q in texts]
        return [(i, np.ones(len(i))) for i in ids]


def _hash(token):
    import mmh3

    return abs(mmh3.hash(token))


class FastembedSparse(Sparse):
    def __init__(self, model, idf):
        from fastembed import SparseTextEmbedding

        self.model_name, self.idf = model, idf
        self.name = "fastembed-" + model.replace("/", "_")
        self._model = None
        self._cls = SparseTextEmbedding

    @property
    def model(self):
        if self._model is None:
            self._model = self._cls(self.model_name, threads=os.cpu_count())
        return self._model

    def encode_docs(self, texts):
        return [(e.indices, e.values) for e in self.model.embed(list(texts), batch_size=32)]

    def encode_queries(self, texts):
        return [(e.indices, e.values) for e in self.model.query_embed(list(texts))]


class BM25s:
    """bm25s with PyStemmer English stemming and English stopwords (Lucene-like)."""

    name = "bm25s"

    def fit(self, corpus_name, texts):
        import bm25s
        import Stemmer

        self.stemmer = Stemmer.Stemmer("english")
        self.bm25s = bm25s
        tokens = bm25s.tokenize(list(texts), stopwords="en", stemmer=self.stemmer, show_progress=False)
        self.retriever = bm25s.BM25(k1=1.2, b=0.75)
        self.retriever.index(tokens, show_progress=False)
        self.n = len(texts)
        return self

    def scores(self, queries):
        out = np.zeros((len(queries), self.n), dtype=np.float32)
        for i, q in enumerate(queries):
            toks = self.bm25s.tokenize([q], stopwords="en", stemmer=self.stemmer, show_progress=False)
            ids = toks.ids[0]
            # map query token ids into the index vocabulary
            inv = {v: k for k, v in toks.vocab.items()}
            words = [inv[t] for t in ids]
            vocab = self.retriever.vocab_dict
            qids = [vocab[w] for w in words if w in vocab]
            if qids:
                out[i] = self.retriever.get_scores(qids)
        return out


class BGEM3Sparse(Sparse):
    idf = False
    name = "bge-m3-sparse"

    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, devices="cpu")
        return self._model

    def _encode(self, texts, batch):
        out = self.model.encode(list(texts), batch_size=batch, max_length=512, return_dense=False, return_sparse=True)
        return [(np.array([int(k) for k in w]), np.array(list(w.values()), dtype=np.float32)) for w in out["lexical_weights"]]

    def encode_docs(self, texts):
        return self._encode(texts, 16)

    def encode_queries(self, texts):
        return self._encode(texts, 64)


class Dense:
    def __init__(self, name, backend, model, query_prefix="", doc_prefix="", prompt_name=None):
        self.name, self.backend, self.model_name = name, backend, model
        self.query_prefix, self.doc_prefix, self.prompt_name = query_prefix, doc_prefix, prompt_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.backend == "fastembed":
                from fastembed import TextEmbedding

                self._model = TextEmbedding(self.model_name, threads=os.cpu_count())
            elif self.backend == "specter2":
                self._model = _Specter2()
            else:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name, device="cpu", trust_remote_code=True)
                self._model.max_seq_length = 512
        return self._model

    def _embed(self, texts, query):
        prefix = self.query_prefix if query else self.doc_prefix
        texts = [prefix + t for t in texts]
        if self.backend == "fastembed":
            vecs = np.array(list(self.model.embed(texts, batch_size=32)))
        elif self.backend == "specter2":
            vecs = self.model.encode(texts, query)
        else:
            kw = {"prompt_name": self.prompt_name} if (query and self.prompt_name) else {}
            vecs = self.model.encode(texts, batch_size=16, convert_to_numpy=True, **kw)
        return (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)

    def fit(self, corpus_name, texts):
        path = CACHE / self.name / f"{corpus_name}.npy"
        if path.exists():
            self.docs = np.load(path)
        else:
            self.docs = self._embed(list(texts), query=False)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, self.docs)
        return self

    def scores(self, queries):
        return self._embed(list(queries), query=True) @ self.docs.T


class _Specter2:
    """SPECTER2 with the proximity adapter for papers and the ad-hoc query adapter for queries."""

    def __init__(self):
        import torch
        from adapters import AutoAdapterModel
        from transformers import AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
        self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=False)
        self.model.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="adhoc_query", set_active=False)
        self.model.eval()

    def encode(self, texts, query):
        self.model.set_active_adapters("adhoc_query" if query else "proximity")
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), 16):
                batch = self.tok(texts[i : i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt")
                out.append(self.model(**batch).last_hidden_state[:, 0, :].numpy())
        return np.concatenate(out)


class Hybrid:
    """Reciprocal rank fusion of two systems (k = 60, top 1000 of each)."""

    def __init__(self, a, b, k=60, depth=1000):
        self.a, self.b, self.k, self.depth = a, b, k, depth
        self.name = f"rrf({a.name}+{b.name})"

    def fit(self, corpus_name, texts):
        self.a.fit(corpus_name, texts)
        self.b.fit(corpus_name, texts)
        return self

    def scores(self, queries):
        out = None
        for s in (self.a, self.b):
            sc = s.scores(queries)
            ranks = np.argsort(-sc, axis=1)[:, : self.depth]
            fused = np.zeros_like(sc)
            rows = np.arange(len(queries))[:, None]
            fused[rows, ranks] = 1.0 / (self.k + 1 + np.arange(ranks.shape[1]))[None, :]
            out = fused if out is None else out + fused
        return out


def baselines():
    """All non-scitok systems, by name (constructed lazily: some load large models)."""
    return {
        "whitespace-bm25": lambda: Whitespace(),
        "bm25s": lambda: BM25s(),
        "fastembed-bm25": lambda: FastembedSparse("Qdrant/bm25", idf=True),
        "bm42": lambda: FastembedSparse("Qdrant/bm42-all-minilm-l6-v2-attentions", idf=True),
        "minicoil": lambda: FastembedSparse("Qdrant/minicoil-v1", idf=True),
        "splade-pp": lambda: FastembedSparse("prithivida/Splade_PP_en_v1", idf=False),
        "opensearch-sparse-v3": lambda: FastembedSparse("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte", idf=False),
        "bge-m3-sparse": lambda: BGEM3Sparse(),
        "bge-base": lambda: Dense("bge-base", "fastembed", "BAAI/bge-base-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: "),
        "arctic-m": lambda: Dense("arctic-m", "fastembed", "snowflake/snowflake-arctic-embed-m", query_prefix="Represent this sentence for searching relevant passages: "),
        "e5-base": lambda: Dense("e5-base", "st", "intfloat/e5-base-v2", query_prefix="query: ", doc_prefix="passage: "),
        "specter2": lambda: Dense("specter2", "specter2", "allenai/specter2"),
        "qwen3-0.6b": lambda: Dense("qwen3-0.6b", "st", "Qwen/Qwen3-Embedding-0.6B", prompt_name="query"),
    }
