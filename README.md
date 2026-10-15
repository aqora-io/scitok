# scitok

A fast, domain-aware tokenizer for scientific papers that produces sparse vectors for
[Qdrant](https://qdrant.tech) collections using server-side IDF (`modifier: idf`). It ships
with a quantum physics / quantum computing (`quant-ph`) domain, and a learning binary,
`scitok-learn`, builds domains for any arXiv category from recent papers.

What it does differently from a generic BM25 tokenizer:

| Input | Terms |
| --- | --- |
| `We run QAOA` | `run` `~qaoa` |
| `the quantum approximate optimization algorithm` | `~qaoa` `quantum` `approxim` `optim` `algorithm` |
| `Lieb–Robinson bounds` / `Lieb--Robinson bound` | `~lieb_robinson_bound` `lieb` `liebrobinson` `robinson` `bound` |
| `Lieb Robinson bound` | `~lieb_robinson_bound` `lieb` `robinson` `bound` |
| `Schr\"odinger cat states` (LaTeX) / `Schrödinger cat states` | `~cat_state` `schroding` `cat` `state` |
| `$O(\sqrt{N})$ queries via Grover's algorithm \cite{g96}` | `queri` `~grover_algorithm` `grover` `algorithm` |
| `the frobnicated quantum widget (FQW) ... the FQW` | later `FQW`s also count toward `frobnic` `quantum` `widget` |
| `ETH` vs `eth` (case-sensitive acronym) | `~eth` vs `eth` |

- **Concept terms** (`~…`): acronyms, spelling variants and long forms map to one term. Phrase
  matches emit the concept *and* the constituent words, so you get the phrase's precision and
  the words' recall.
- **LaTeX**: inline math in abstracts and full `.tex` sources. Citations, references, floats
  (except captions), comments and math are dropped; `SU(2)`, `\mathbb{Z}_2`, `$T$-gate` survive.
- **Normalization**: ligatures, diacritics, Greek letters, dashes (including LaTeX `--`),
  possessives, plurals and hyphen/space variants.
- **In-document acronyms**: `long form (LF)` definitions are detected with
  Schwartz–Hearst, and later uses of `LF` are expanded.
- **Stable term ids**: `abs(murmur3_32(term))`, the same convention as fastembed's
  `Qdrant/bm25`. There is no vocabulary file.

## Python

```bash
maturin develop --release          # or: maturin build --release
```

```python
from qdrant_client import QdrantClient, models
import scitok

tok = scitok.Tokenizer()  # built-in quant-ph domain, BM25 weighting
client = QdrantClient(url="http://localhost:6333")
client.create_collection(
    "papers",
    vectors_config={},
    sparse_vectors_config={"text": models.SparseVectorParams(modifier=models.Modifier.IDF)},
)

docs = [f"{p['title']}.\n{p['abstract']}" for p in papers]
client.upsert("papers", [
    models.PointStruct(id=i, vector={"text": models.SparseVector(indices=idx, values=val)})
    for i, (idx, val) in enumerate(tok.embed_documents(docs))  # parallel, releases the GIL
])

idx, val = tok.embed_query("QAOA MaxCut")
hits = client.query_points("papers", query=models.SparseVector(indices=idx, values=val), using="text")
```

Options: `Tokenizer(domains=["quant-ph"], domain_paths=["path/to/domain"], input="auto"|"plain"|"latex",
weighting="bm25"|"tf", k1=1.2, b=0.75, avg_len=None)`. Also available: `tok.tokenize(text)` for
inspection, `scitok.builtin_domains()` and `scitok.term_id(term)`.

Use `input="latex"` for full LaTeX documents. Its BM25 `avg_len` defaults to the domain's
full-text average when that exists. Keep abstracts and full texts in separate sparse vectors.

## Rust

```rust
use scitok::{Config, Domain, Input, Tokenizer, Weighting};

let tok = Tokenizer::new(Config::default());
let doc = tok.embed_document("Grover's algorithm on a surface code"); // SparseVector { indices, values }
let query = tok.embed_query("surface code threshold");

let custom = Tokenizer::new(Config {
    domains: vec![Domain::builtin("quant-ph")?, Domain::load("domains/cond-mat.str-el")?],
    input: Input::Latex,
    weighting: Weighting::Bm25 { k1: 1.2, b: 0.75, avg_len: None },
});
```

`cargo run --example tokenize -- "some text"` prints the terms for each argument.

## Domains

A domain is a directory named after an arXiv category:

| File | Written by | Purpose |
| --- | --- | --- |
| `curated.toml` | hand | Lexicon. Always wins over mined entries. |
| `mined.toml` | `scitok-learn` | Lexicon: acronyms, eponyms, collocations. Regenerated, reviewed in diffs. |
| `stats.toml` | `scitok-learn` | Corpus size, date range, average document length (BM25 `avg_len`). |
| `stopwords.txt` | hand | Extra stopwords (optional). |
| `stopword_candidates.txt` | `scitok-learn` | Very high document-frequency words, for review only. Not loaded. |

Lexicon entries:

```toml
version = 1

[[term]]
canonical = "qaoa"            # emitted as ~qaoa
variants = ["QAOA", "quantum approximate optimization algorithm", "quantum approximate optimisation algorithm"]

[[term]]
canonical = "eth"
variants = ["ETH", "eigenstate thermalization hypothesis"]
case_sensitive = true         # acronym variants must match case ("eth" stays a plain word)
```

Variants are matched case-insensitively and ignore plurals, possessives and hyphen/space
differences, so list only genuinely different spellings.

When several domains are loaded, the curated entries of all domains come first, then the mined
ones. If two domains define the same variant, the first domain listed wins.

Directories under `crates/scitok/data/domains/` are compiled into the library behind a
`domain-<name>` cargo feature. `quant-ph` is the default. To add a built-in domain, run
`scitok-learn`, commit the directory, and add the feature to `crates/scitok/Cargo.toml`.

> **Changing domain data changes term ids.** Any change to a lexicon, stopwords or tokenizer
> version changes the terms a document is indexed under. Re-index the collection afterwards.
> Domain files carry a format `version`, and mismatches are rejected.

## Learning a domain: `scitok-learn`

```bash
cargo run --release -p scitok-learn -- learn --category quant-ph            # last 6 months, abstracts
cargo run --release -p scitok-learn -- learn --category cond-mat.str-el \
    --from 2026-01-01 --until 2026-06-30 --full-text --contact you@example.org
```

- `fetch` harvests arXiv's OAI-PMH endpoint into `.scitok-cache/<category>/`. With `--full-text`
  it also downloads e-print LaTeX sources (main file found, `\input`s inlined, PDF-only skipped).
- `build` mines the cache offline into the domain directory. The default is
  `crates/scitok/data/domains/<category>`; change it with `--out`. Iterate on thresholds with
  `--min-count`, `--min-npmi`, `--max-phrases` and `--max-ngram`.
- `learn` runs `fetch` then `build`.

Harvesting is polite and resumable:
- The default rate is `--rate 1/3s`, arXiv's policy. Faster rates print a warning.
- `Retry-After` and backoff are honored.
- The resumption token is saved after every page, so an interrupted run (or one stopped by
  `--max-papers`) continues where it left off.

The date range selects papers *first submitted* in the range. Pass `--include-revisions` to also
include older papers revised in it. Only papers whose primary category is the requested one are
kept, unless you pass `--include-cross-lists`. For large full-text corpora, use arXiv's
[S3 bulk data](https://info.arxiv.org/help/bulk_data_s3.html) instead of `--full-text`.

Mining runs on the same tokens as runtime tokenization, against the curated lexicon only:
- Acronym ↔ long-form pairs: Schwartz–Hearst over the corpus, keeping pairs where one long form
  clearly dominates.
- Eponyms: `Hatano-Nelson model`, `Leggett-Garg inequality`.
- 2–4-word collocations: ranked by NPMI × log document frequency, with generic academic phrases
  filtered out.

The shipped `quant-ph` domain was built from the 9,264 quant-ph papers submitted between
2026-04-15 and 2026-10-15 (abstracts only). It combines 364 curated and about 1,700 mined terms.

## Performance

Measured on the 9,264 quant-ph abstracts (12.6 MB), embedding to BM25 sparse vectors:

| | Throughput |
| --- | --- |
| single thread | ~21 MB/s (~15k abstracts/s) |
| `embed_documents`, 12 threads | ~195 MB/s |
| fastembed `Qdrant/bm25`, same corpus | ~2.8 MB/s |

Reproduce with `SCITOK_BENCH_CORPUS=abstracts.txt cargo bench -p scitok` (one document per line)
or `cargo run --release --example embed_file -- abstracts.txt`.

## Development

```bash
cargo test --workspace
maturin develop && pytest tests/python
```
