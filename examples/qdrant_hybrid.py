"""Hybrid search over arXiv abstracts with Qdrant: scitok sparse vectors plus a dense model,
fused with reciprocal rank fusion (the best setup in eval/RESULTS.md). Queries are expanded for
the dense model, `QAOA` → `QAOA (quantum approximate optimization algorithm)`, which helps when a
query and a paper name a concept differently.

    pip install qdrant-client fastembed scitok    # or `maturin develop` for scitok
    docker run -p 6333:6333 qdrant/qdrant
    python examples/qdrant_hybrid.py .scitok-cache/quant-ph/abstracts.jsonl "QAOA for MaxCut" "Lieb-Robinson bounds"

Papers are JSON lines with `title` and `abstract`, the format of scitok-learn's cache.
Term ids change with scitok's lexicon, so re-index after upgrading scitok.
"""

import argparse
import json
from itertools import islice

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

import scitok

DENSE_MODEL = "snowflake/snowflake-arctic-embed-m"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def index(client, collection, papers, tok, dense, batch=256):
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=768, distance=models.Distance.COSINE)},
        # scitok emits BM25 term frequencies; Qdrant supplies the IDF
        sparse_vectors_config={"scitok": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    for start in range(0, len(papers), batch):
        chunk = papers[start : start + batch]
        texts = [f"{p['title']}.\n{p['abstract']}" for p in chunk]
        client.upsert(collection, [
            models.PointStruct(
                id=start + i,
                vector={"dense": d.tolist(), "scitok": models.SparseVector(indices=idx, values=val)},
                payload=p,
            )
            # fastembed's default batch of 256 abstracts needs over 10 GB of RAM; 16 needs about 1.3 GB
            for i, (p, (idx, val), d) in enumerate(zip(chunk, tok.embed_documents(texts), dense.embed(texts, batch_size=16)))
        ])


def search(client, collection, query, tok, dense, limit=10):
    idx, val = tok.embed_query(query)  # don't lowercase queries: acronyms like ETH are case-sensitive
    return client.query_points(
        collection,
        prefetch=[
            models.Prefetch(query=models.SparseVector(indices=idx, values=val), using="scitok", limit=100),
            models.Prefetch(query=next(dense.embed([QUERY_PREFIX + tok.expand(query)])).tolist(), using="dense", limit=100),
        ],
        # k=61 in Qdrant is the k=60 of the RRF paper and the evaluation; Qdrant's default is 2
        query=models.RrfQuery(rrf=models.Rrf(k=61)),
        limit=limit,
    ).points


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("papers", help="JSON lines with title and abstract")
    ap.add_argument("queries", nargs="+")
    ap.add_argument("--url", default="http://localhost:6333", help="Qdrant URL, or :memory:")
    ap.add_argument("--collection", default="scitok_example")
    ap.add_argument("--max-docs", type=int, default=2000, help="the dense model is slow on CPU")
    a = ap.parse_args()

    with open(a.papers) as f:
        papers = [json.loads(line) for line in islice(f, a.max_docs)]
    tok = scitok.Tokenizer()  # built-in quant-ph domain
    dense = TextEmbedding(DENSE_MODEL)
    client = QdrantClient(location=a.url)
    index(client, a.collection, papers, tok, dense)
    for q in a.queries:
        print(f"\n{q}")
        for hit in search(client, a.collection, q, tok, dense):
            print(f"  {hit.score:.3f}  {hit.payload.get('id', '')}  {hit.payload['title']}")


if __name__ == "__main__":
    main()
