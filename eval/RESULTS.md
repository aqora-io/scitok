# scitok evaluation on quant-ph

Evaluation of scitok against lexical, learned-sparse and dense retrievers on recent arXiv
quant-ph papers, and of scitok's acronym detector against scispaCy and the reference
Schwartz–Hearst implementation. All numbers below are on **held-out test splits** unless
marked dev. Tuning used dev splits only.

## Setup

- **Corpus:** 9,264 primary quant-ph papers first submitted 2026-04-15 – 2026-10-15
  (title + abstract; the title→paper tasks index abstracts only).
- **No leakage from mining:** the evaluated scitok domain was mined only from the *previous*
  six months (4,480 papers, arXiv ids 2510–2603). The curated lexicon predates the evaluation.
- **Scoring:** every sparse system is scored like Qdrant does: a dot product, with Qdrant's IDF
  modifier `ln(1 + (N − n + 0.5)/(n + 0.5))` for the BM25-style systems. Dense models use cosine
  similarity. Hybrids use reciprocal rank fusion (k = 60).
- **Splits:** by source paper, about 40% dev / 60% test, seeded.

### Tasks

| task | queries (test) | what it measures |
|---|---|---|
| **agent** | 170 (150 dev) | Realistic search: subagents wrote a keyword query and a natural-language question for 160 sampled papers, without copying titles. Graded relevance (0/1/2) on pooled top-10 results of 13 systems (11,852 judgments), judged blind. |
| title | 5,531 | Known-item: title → its abstract. |
| acronym | 928 | Title with an acronym ↔ long form swapped, where the abstract defines both. Easy. |
| acronym_hard | 568 | As above, but the swapped-in form never occurs in the target abstract. Pairs come from the reference S–H implementation over the mining corpus, not from scitok. |

How the agent-query judgments were made:
- Each judge saw two queries, each with about 37 shuffled candidates under opaque ids, and had to
  read every abstract in full.
- A first judging round was discarded: query ids leaked the source paper, and batches were too
  large for full reading.
- Sanity check: although the judges could not identify it, the source paper was graded 2 for 320 of 320 queries.
- Labels: a median of 9 relevant documents per query; 100% of every evaluated system's top 10 is judged.

## Results

### Realistic queries (agent, test, n = 170)

| system | nDCG@10 | MRR@10 | R@100 |
|---|---|---|---|
| RRF(**scitok tuned** + arctic-m) | **0.764** | 0.974 | **0.975** |
| RRF(fastembed BM25 + arctic-m) | 0.761 | 0.962 | **0.975** |
| arctic-m (dense, snowflake-arctic-embed-m) | 0.736 | **0.979** | 0.895 |
| opensearch-neural-sparse-v3 (gte) | 0.729 | 0.950 | 0.908 |
| miniCOIL v1 | 0.723 | 0.956 | 0.910 |
| **scitok (tuned)** | 0.721 | 0.966 | **0.928** |
| bm25s (Snowball, stopwords) | 0.718 | 0.961 | 0.917 |
| fastembed `Qdrant/bm25` | 0.716 | 0.961 | 0.914 |
| bge-base-en-v1.5 | 0.710 | 0.952 | 0.868 |
| SPLADE++ (prithivida) | 0.706 | 0.950 | 0.888 |
| whitespace BM25 | 0.697 | 0.957 | 0.872 |
| bm42 | 0.678 | 0.934 | 0.863 |
| scitok (original defaults) | 0.678 | 0.926 | 0.899 |
| e5-base-v2 | 0.633 | 0.911 | 0.820 |

Paired bootstrap on nDCG@10 (10k resamples):

| comparison | Δ | 95% CI |
|---|---|---|
| scitok tuned − scitok original | **+0.043** | [+0.023, +0.064] |
| scitok tuned − fastembed BM25 | +0.004 | [−0.009, +0.017] |
| scitok tuned − bm25s | +0.003 | [−0.010, +0.017] |
| scitok tuned − opensearch-sparse-v3 | −0.008 | [−0.035, +0.019] |
| scitok tuned − arctic-m | −0.015 | [−0.040, +0.011] |
| RRF(scitok + arctic-m) − arctic-m | **+0.028** | [+0.013, +0.044] |
| RRF(scitok + arctic-m) − RRF(fastembed + arctic-m) | +0.001 | [−0.006, +0.007] |

### Proxy tasks (test)

| system | title MRR@10 | acronym MRR@10 | acronym_hard MRR@10 | acronym_hard S@10 |
|---|---|---|---|---|
| **scitok (tuned)** | **0.897** | **0.962** | 0.772 | **0.896** |
| scitok (original defaults) | 0.878 | 0.951 | **0.779** | 0.884 |
| bm25s | 0.894 | 0.947 | 0.727 | 0.868 |
| fastembed `Qdrant/bm25` | 0.892 | 0.949 | 0.718 | 0.850 |
| whitespace BM25 | 0.875 | 0.932 | 0.713 | 0.870 |

Paired bootstrap, tuned scitok − fastembed BM25 (reciprocal rank):

| task | Δ | 95% CI |
|---|---|---|
| title | +0.005 | [+0.001, +0.008] |
| acronym_hard | **+0.054** | [+0.036, +0.072] |

### Acronym detection

Precision, recall and F1 of (short form, long form) definitions:
- Gold: agent-annotated abstracts.
- Strict: the long form must match exactly after normalizing case, hyphens, diacritics and possessives.
- Lenient: the long form must overlap at least half of the gold words.

| detector | dev P / R / F1 (strict, n = 225) | held-out P / R / F1 (strict, n = 99) | held-out F1 (lenient) |
|---|---|---|---|
| **scitok** | 0.981 / 0.938 / **0.959** | 0.978 / 0.889 / **0.931** | 0.952 |
| `abbreviations` (reference Schwartz–Hearst) | 0.959 / 0.929 / 0.944 | 0.937 / 0.899 / 0.918 | **0.979** |
| scispaCy `AbbreviationDetector` | 0.958 / 0.916 / 0.936 | 0.947 / 0.899 / 0.922 | 0.953 |

scitok's detector was improved after inspecting errors on the dev set, so the dev numbers
are optimistic. The held-out set was annotated afterwards and never inspected:
- On held-out, scitok has the best precision and strict F1.
- The reference implementation has the best lenient F1, from slightly better recall.
- Differences of about 0.01 on 99 definitions are within noise.

## What tuning changed

Every choice below was made on dev splits:

| change | before | after | why |
|---|---|---|---|
| joined hyphen compounds (`two-qubit` → `twoqubit`) | on | **off** | They match unrelated papers using the same compound, and they give hyphenated words three query terms. Costs about 0.01–0.02 everywhere. |
| weight of multi-word concept terms | 1.0 | **0.5** | Phrases in a lexicon double-count their words. At 1.0 they hurt title search and realistic queries. At 0 the acronym gains vanish, because long forms map to the acronym's concept. 0.5 was best on the agent dev set (0.748 vs 0.699 at 1.0). |
| acronym short forms | single all-caps token | also hyphenated (`BP-OSD`) and digit-bearing (`TCL6`) forms | Better recall in acronym detection. |
| acronym long forms | could start inside a compound | extend to the compound start | `out-of-time-order correlators`, not `of time order correlators`. |
| mining `min_count` | 5 | 3 at 4.5k abstracts, 5 for the shipped 13.7k | More acronym pairs; collocations matter less. |
| BM25 k1, b; stopword list | | unchanged | k1/b were flat. NLTK stopwords gave no gain over scitok's list. |

The single-word part of the original concept weighting was kept. Acronym matches replace their
unigram and always count fully.

## LLM-refined lexicon

`scitok-learn refine` was run with `gpt-6.1-sol` on the tuned eval domain. It saw only the
4,480 mining-period abstracts: snippets and variant attestation come from that corpus, and none
of the evaluation-period text was used.

| step | result |
|---|---|
| review of 1,640 mined terms | 1,060 kept (405 renamed), 517 dropped, 63 merged |
| expansion of 1,424 concepts | 1,475 variants proposed, 252 accepted |
| rejected proposals | 950 not in ≥ 2 mining documents, 199 claimed by another concept, 47 too broad, 27 already known |
| cost | 89 requests, 376k prompt + 128k completion tokens |

A spot check of random decisions found them sensible:
- Drops are generic phrases: `practical_guidance`, `attracted_considerable_attention`.
- Merges join duplicates: `barren_plateaus` → `barren_plateau`, `gksl` → `lindblad_master_equation`,
  `twin_field` → `tf_qkd`.
- Added variants include `stabiliser code`, `MaxCut`, `SiV` and `GME`. A few broad acronyms
  (`AI`, `LP`, `TN`) also passed the guards.

The base lexicon's `stats.toml` length was computed before compounds were turned off. Each
lexicon below therefore uses the BM25 `avg_len` recomputed on the mining corpus with the current
tokenizer: 117.7 for mined, 115.9 for refined. The ablations split `reviewed.toml` into its review
part and its added variants.

Test results:

| lexicon | agent nDCG@10 | title MRR@10 | acronym MRR@10 | acronym_hard MRR@10 |
|---|---|---|---|---|
| mined (scitok tuned, as above) | 0.721 | 0.897 | 0.962 | 0.772 |
| mined, recomputed `avg_len` | 0.720 | 0.896 | 0.962 | 0.773 |
| **LLM review + expand** | **0.723** | **0.898** | **0.963** | **0.777** |
| review only | 0.722 | 0.897 | **0.963** | 0.772 |
| expand only | 0.722 | 0.896 | **0.963** | **0.777** |

Paired bootstrap, LLM review + expand − mined (recomputed `avg_len`):

| task | Δ | 95% CI | queries changed |
|---|---|---|---|
| agent (nDCG@10) | +0.003 | [−0.001, +0.008] | 45 / 170 |
| title | +0.001 | [+0.000, +0.002] | 108 / 5,531 |
| acronym | +0.001 | [−0.002, +0.004] | 11 / 928 |
| acronym_hard | +0.004 | [−0.001, +0.011] | 28 / 568 |

Against fastembed BM25, the refined lexicon gains on every task:
- agent: +0.007 [−0.007, +0.020];
- title: +0.006 [+0.002, +0.009];
- acronym_hard: +0.059 [+0.040, +0.078].

Dev agreed with test: agent ±0.001 and acronym_hard +0.005. The judged pool still covers
0.998 of the refined lexicon's test top 10 (dev 0.993), so unjudged documents do not matter.

What the refinement did:
- It moves every task in the right direction, but by small amounts. Only title clears zero, and
  only barely.
- The gains are small because the miner already finds the acronym pairs these tasks rely on.
  Most of what the LLM adds, such as other spellings and notation, rarely separates a query from
  its relevant papers in this corpus.
- Expansion produces the acronym_hard gain, as expected from new acronym ↔ long-form pairs.
- Review removes noise. Dropping 517 generic phrases changes 35 of 170 agent rankings, for
  +0.002.
- It never made a task worse. It also leaves a cleaner lexicon with readable names and
  auditable reasons, so the shipped domain now uses a refined `reviewed.toml`.

### Discovering missed concepts

A third `refine` pass, Discover, gave the model 2,000 sampled mining-period abstracts. It ran in
batches of 25, each with the concepts the lexicon already finds in each abstract. The model
listed the domain concepts the lexicon lacked. The same code guards as Expand applied, and
discovered concepts were then expanded too.

| step | result |
|---|---|
| proposals | 6,340 concepts |
| added | 1,515 concepts with 1,737 variants; median document frequency 3 |
| rejected variants | 4,278 in fewer than 2 mining documents, 309 already in the lexicon, 433 single words or lone acronyms |
| expansion | 340 variants accepted, now including the discovered concepts |
| cost | 131 requests (88 cached), 572k prompt + 307k completion tokens, 88 minutes |

Most of a random sample of 30 were real concepts, for example:
- `pauli_noise`, `quantum_trajectory`, `cornell_potential`;
- `spin_lock_induced_crossing` (SLIC), `entangled_two_photon_absorption` (ETPA).

About 1 in 10 were weak: `niobium` ← "niobium-based", "Grover-based heuristic" as `grover_search`,
and "bipartite and tripartite entanglements".

Test results (judged@10 coverage 0.992):

| lexicon | agent nDCG@10 | title MRR@10 | acronym MRR@10 | acronym_hard MRR@10 |
|---|---|---|---|---|
| mined, recomputed `avg_len` | 0.720 | 0.896 | 0.962 | 0.773 |
| LLM review + expand | 0.723 | 0.898 | **0.963** | **0.777** |
| **LLM review + discover + expand** | **0.729** | **0.901** | **0.963** | 0.775 |

Paired bootstrap:

| comparison | agent Δ | title Δ |
|---|---|---|
| + discover − review + expand | **+0.006** [+0.000, +0.012] | **+0.003** [+0.001, +0.005] |
| + discover − mined | **+0.009** [+0.002, +0.017] | **+0.004** [+0.002, +0.006] |
| + discover − fastembed BM25 | +0.013 [−0.001, +0.027] | **+0.009** [+0.005, +0.013] |

The acronym tasks did not change: −0.001 [−0.003, +0.001] and −0.002 [−0.008, +0.004]. Their
acronym pairs were already in the lexicon. Dev agreed in direction but was smaller: agent +0.001,
title +0.004, acronym_hard −0.002.

Discovery is the first lexicon change with a clear gain on realistic queries. scitok's agent
nDCG@10 of 0.729 now ties the best learned sparse model (opensearch-neural-sparse-v3, 0.729).

## Dense input: raw, cleaned or expanded text

scitok can rewrite the text a dense model embeds:
- `clean`: LaTeX converted to text, with math dropped, and Unicode normalized.
- `expand`: cleaned text with each lexicon concept's preferred name in parentheses after its first
  mention, e.g. `QAOA (quantum approximate optimization algorithm)`. Concepts that the text
  already calls by that name, or defines as an acronym, are left alone.
- `expand` with `input="plain"`: the same, but the LaTeX, math included, is kept.

Setup:
- arctic-m embedded each variant, for documents and queries alike, and also for queries only.
  Queries-only keeps the raw document embeddings.
- Expansion used the tuned eval lexicon, so there is no leakage, or its LLM-refined version
  (review + discover + expand).
- The sparse half of every hybrid is tuned scitok.
- About a third of the abstracts get at least one insertion. Of the queries, 2–3% contain LaTeX.

Realistic queries, nDCG@10, as arctic-m alone / RRF with scitok:

| dense input | dev (n = 150) | test (n = 170) |
|---|---|---|
| raw | 0.712 / 0.767 | 0.736 / 0.764 |
| clean | 0.705 / 0.763 | 0.738 / 0.764 |
| expand | 0.710 / 0.767 | 0.737 / **0.768** |
| expand, math kept | **0.715** / 0.766 | 0.732 / 0.767 |
| expand, LLM-refined lexicon | 0.710 / 0.765 | 0.737 / 0.767 |
| expand, queries only | 0.714 / **0.768** | 0.735 / **0.768** |

None of the variants differs significantly from raw on test (paired bootstrap). Every hybrid is
within 0.004 of raw.

The proxy tasks were run for raw and for two expansion settings:
- **expand with math kept, on documents and queries.** It was chosen on agent dev, where it was the
  best dense-only system.
- **expand on queries only**, which needs no document re-encoding.

Results are MRR@10 on test, with dev in parentheses:

| system | acronym | acronym_hard | title |
|---|---|---|---|
| arctic-m, raw | 0.947 (0.945) | 0.832 (0.821) | 0.936 (0.938) |
| arctic-m, expand documents + queries | 0.950 (0.940) | **0.864** (0.862) | 0.934 (0.937) |
| arctic-m, expand queries only | 0.949 (0.945) | 0.859 (0.854) | 0.933 (0.935) |
| RRF, raw | 0.965 (0.973) | 0.838 (0.842) | 0.932 (0.930) |
| RRF, expand documents + queries | 0.964 (0.969) | 0.848 (0.853) | 0.931 (0.930) |
| RRF, expand queries only | 0.965 (0.970) | **0.852** (0.853) | 0.931 (0.930) |

Paired bootstrap on acronym_hard, expand − raw:

| system | dev | test |
|---|---|---|
| arctic-m, documents + queries | **+0.041** [+0.024, +0.059] | **+0.032** [+0.017, +0.048] |
| arctic-m, queries only | **+0.032** [+0.016, +0.050] | **+0.027** [+0.013, +0.042] |
| RRF, documents + queries | +0.011 [−0.003, +0.024] | +0.009 [−0.003, +0.022] |
| RRF, queries only | +0.011 [−0.001, +0.023] | **+0.014** [+0.003, +0.026] |

On acronym the differences are within ±0.005 and every CI includes 0. On title they are within
0.003.

Conclusions:
- Expansion helps where a query and a paper name a concept differently and the paper never spells
  out the query's form. arctic-m gains 0.03–0.04 MRR there.
- Expanding only the queries keeps most of that gain for arctic-m. In the hybrid it does at least
  as well as expanding documents too: +0.014 on test, the only significant hybrid gain.
- The hybrid gains less than arctic-m alone, because scitok's sparse side already bridges those
  pairs.
- Elsewhere, expansion is neutral.
- Dropping math does not help.
- The LLM-refined lexicon performs the same as the mined one.

[examples/qdrant_hybrid.py](../examples/qdrant_hybrid.py) expands only the queries for the dense
model.

## Findings

1. On realistic topical queries, tuned scitok is **statistically tied** with the best BM25
   implementations and with learned sparse models (opensearch-v3, miniCOIL, SPLADE++) on nDCG@10.
   It has the **best recall@100 among single systems** (0.928).
2. scitok is clearly better than every BM25 baseline when **queries and papers use different
   names for the same concept**: +0.054 MRR on acronym-swapped titles. That is what the
   lexicon and in-document acronym expansion are for. It is also slightly but significantly
   better on title known-item search.
3. Combined with a dense model via RRF, scitok gives the best overall system (0.764
   nDCG@10, 0.975 R@100). It ties a fastembed BM25 hybrid on nDCG and has higher MRR.
   **For production, use scitok together with a dense embedding.**
4. On this corpus the dense models are not dominant. arctic-m is the best single system on nDCG
   and MRR, while e5-base and bge-base trail BM25. The queries were written by agents that had read the
   abstracts, which likely favors lexical overlap.
5. scitok embeds about 17× faster than fastembed BM25 on CPU. It is orders of magnitude faster than
   learned sparse or dense encoders: SPLADE++ took 70 minutes for 9k abstracts on this machine.
6. Reviewing and expanding the mined lexicon with an LLM (`gpt-6.1-sol`) gives small but
   consistent gains: +0.001 to +0.005 on every task, none clearly significant. Letting the LLM
   discover concepts the miner missed is a significant improvement on realistic queries (+0.009
   nDCG@10 over the mined lexicon) and on title search. This brings scitok level with the best
   learned sparse model.
7. Expanding queries with scitok before the dense model sees them (`tok.expand(query)`)
   improves retrieval on acronym-swapped titles: +0.027 MRR for arctic-m on test and +0.014 in
   the hybrid. Expanding documents as well adds little. Other tasks do not change, and converting
   LaTeX to text alone does not help.

## Limitations

- Relevance labels come from LLM judges (Claude Sonnet), not domain experts. There is one judge
  per query, so inter-annotator agreement was not measured.
- Systems that were not in the judging pool (SPECTER2, BGE-M3, Qwen3-Embedding) are still
  encoding on CPU and are not reported. When they are added, unjudged documents count as
  non-relevant, which slightly disadvantages them.
- Proxy tasks are reported for lexical systems, and for arctic-m in
  [Dense input](#dense-input-raw-cleaned-or-expanded-text). The other dense and learned-sparse
  encodings of the abstract-only corpus were still running.
- 170 test queries resolve differences of about 0.02 nDCG; smaller differences are noise.
- The shipped built-in domain is mined (and LLM-refined) from 12 months, including the evaluation
  period. It was not evaluated, to avoid leakage.
- The LLM-refined lexicon was evaluated once per split with one model and one run. Different
  models or reruns would produce somewhat different lexicons.

## Reproduce

See `eval/README.md`.
