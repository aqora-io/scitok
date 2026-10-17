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
| `Lieb–Robinson bounds` / `Lieb--Robinson bound` / `Lieb Robinson bound` | `~lieb_robinson_bound` `lieb` `robinson` `bound` |
| `Schr\"odinger cat states` (LaTeX) / `Schrödinger cat states` | `~cat_state` `schroding` `cat` `state` |
| `$O(\sqrt{N})$ queries via Grover's algorithm \cite{g96}` | `queri` `~grover_algorithm` `grover` `algorithm` |
| `the frobnicated quantum widget (FQW) ... the FQW` | later `FQW`s also count toward `frobnic` `quantum` `widget` |
| `ETH` vs `eth` (case-sensitive acronym) | `~eth` vs `eth` |

- **Concept terms** (`~…`): acronyms, spelling variants and long forms map to one term. Phrase
  matches emit the concept at half weight *and* the constituent words, so you get the phrase's
  precision and the words' recall. Acronym matches replace the word and count fully.
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
weighting="bm25"|"tf", k1=1.2, b=0.75, avg_len=None)`. The pipeline switches `concepts`,
`phrase_unigrams`, `compounds`, `stem`, `expand_acronyms` and `concept_weight` default to the
values tuned in the evaluation. Also available: `tok.tokenize(text)` for inspection,
`tok.acronym_definitions(text)`, `scitok.builtin_domains()` and `scitok.term_id(term)`.

For hybrid search, combine scitok with a dense model using Qdrant's RRF fusion. This was the
best setup in the evaluation, and
[examples/qdrant_hybrid.py](https://github.com/aqora-io/scitok/blob/main/examples/qdrant_hybrid.py) shows how. Don't lowercase queries,
because acronyms like `ETH` are case-sensitive. Re-index after upgrading scitok.

scitok can also prepare text for the dense model:
- `tok.clean(text)` returns the text scitok tokenizes: LaTeX converted to text, Unicode normalized.
- `tok.expand(text)` adds each lexicon concept's preferred name after its first mention:
  `QAOA for MaxCut` → `QAOA (quantum approximate optimization algorithm) for MaxCut (maximum cut)`.

The example embeds documents unchanged and the query as `tok.expand(query)`.
Results ([eval/RESULTS.md](https://github.com/aqora-io/scitok/blob/main/eval/RESULTS.md#dense-input-raw-cleaned-or-expanded-text)):
- On acronym-swapped titles, MRR rose by 0.03 for arctic-m and by 0.014 for the hybrid.
- Realistic queries did not change.
- Expanding documents as well added little, and `clean` alone did not help.

Use `input="latex"` for full LaTeX documents. Its BM25 `avg_len` defaults to the domain's
full-text average when that exists. Keep abstracts and full texts in separate sparse vectors.

## Rust

```rust
use scitok::{Config, Domain, Input, Tokenizer, Weighting};

let tok = Tokenizer::new(Config::default());
let doc = tok.embed_document("Grover's algorithm on a surface code"); // SparseVector { indices, values }
let query = tok.embed_query("surface code threshold");

let mut config = Config::default();
config.domains = vec![Domain::builtin("quant-ph")?, Domain::load("domains/cond-mat.str-el")?];
config.input = Input::Latex;
config.weighting = Weighting::Bm25 { k1: 1.2, b: 0.75, avg_len: None };
let custom = Tokenizer::new(config);
```

`cargo run --example tokenize -- "some text"` prints the terms for each argument.

## Domains

A domain is a directory named after an arXiv category:

| File | Written by | Purpose |
| --- | --- | --- |
| `curated.toml` | hand | Lexicon. Always wins over mined entries. |
| `mined.toml` | `scitok-learn` | Lexicon: acronyms, eponyms, collocations. Regenerated, reviewed in diffs. |
| `reviewed.toml` | `scitok-learn refine` | LLM-reviewed `mined.toml` plus added variants (optional). Loaded instead of `mined.toml`. |
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
> Domain files carry a format `version`, and mismatches are rejected. Releases that change the
> terms bump the minor version (`0.1.x` → `0.2.0`); patch releases keep them.

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
- `refine` reviews `build`'s output with an LLM and writes `reviewed.toml` (see below).

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

### LLM review: `refine`

```bash
export OPENAI_API_KEY=...
cargo run --release -p scitok-learn -- refine --category quant-ph --from 2025-10-15 --until 2026-10-15 --dry-run
cargo run --release -p scitok-learn -- refine --category quant-ph --from 2025-10-15 --until 2026-10-15
```

Use the same date range as `build`. `refine` makes three passes over the domain with a chat model
(default `gpt-6.1-sol`; any OpenAI-compatible endpoint via `--base-url`):
- **Review:** each mined term, shown with its document frequency and corpus snippets, is kept
  (optionally under a more readable canonical name), dropped as generic, or merged into another
  concept. Curated entries are never changed.
- **Discover:** the model reads a sample of abstracts (`--discover-docs`, default 2,000; 0 turns it
  off) and lists the concepts in them that the lexicon lacks. These are usually rare, long or
  common-word phrases that the miner's thresholds miss. Only multi-word concepts are added, with
  their acronyms.
- **Expand:** curated, kept and discovered concepts get other names, spellings and acronym ↔ long
  forms.

Proposals from Discover and Expand are checked in code, not trusted:
- a variant is kept only if it occurs in at least `--min-attested` (2) corpus documents;
- variants that already belong to another concept are rejected;
- so is a single plain word, unless every variant of the concept is a single word.

The output is `reviewed.toml`, with every decision and its reason as a comment. The command also
updates `stats.toml`. Responses are cached in `.scitok-cache/<category>/llm/`, so reruns are
free. `--dry-run` counts the requests a run would make. `build` deletes a stale `reviewed.toml`,
so run `refine` again after rebuilding.

The shipped `quant-ph` domain was built from the 13,744 quant-ph papers submitted between
2025-10-15 and 2026-10-15 (abstracts only, `--min-count 5`). It combines 364 curated and 2,229
mined terms. `refine` with `gpt-6.1-sol` then reviewed the mined terms:
- 1,536 kept;
- 84 merged into other concepts;
- 609 dropped as generic;
- 2,390 concepts discovered in 2,000 sampled abstracts, with 2,804 variants;
- 892 variants added to existing and discovered concepts.

## Evaluation

The full evaluation is in [eval/RESULTS.md](https://github.com/aqora-io/scitok/blob/main/eval/RESULTS.md). It was run on 9,264 recent quant-ph
papers against 13 retrievers, with relevance judged blind, and tuned on dev splits only. On the
test splits:
- **Realistic topical queries:** scitok ties the best BM25 implementations and learned sparse
  models (nDCG@10 0.721 vs 0.716–0.729) and has the best single-system recall@100 (0.928).
- **Concept name differs between query and paper:** scitok clearly wins, at +0.054 MRR over
  fastembed BM25 on acronym-swapped queries.
- **Hybrid:** RRF of scitok with a dense model (snowflake-arctic-embed-m) is the best system
  overall (0.764 nDCG@10, 0.975 R@100). Use scitok as the sparse half of a hybrid setup.
- **Acronym detection:** on held-out annotated abstracts, scitok has the highest precision and
  strict F1 (0.931), against scispaCy (0.922) and the reference Schwartz–Hearst implementation
  (0.918).
- **LLM-refined lexicon (`refine`):**
  - Reviewing and expanding the mined lexicon gave small gains on every task (+0.001 to +0.005),
    none clearly significant. It never hurt, and it removed a third of the mined entries as
    generic.
  - Discovering missed concepts added a significant +0.009 nDCG@10 on realistic queries (0.729,
    level with the best learned sparse model).

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
