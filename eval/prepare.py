"""Build the evaluation corpus and proxy query sets from the scitok-learn cache.

Corpus: primary quant-ph papers first submitted 2026-04-15..2026-10-15 (arXiv ids >= 2604).
The eval domain (eval/domains/quant-ph) is mined from the *previous* six months only, so
nothing learned has seen the evaluation documents.

Outputs (eval/data):
  corpus.jsonl            {id, title, abstract}
  queries_title.jsonl     known-item: query = title, target = paper (index abstracts only)
  queries_acronym.jsonl   known-item with acronyms swapped for long forms or vice versa,
                          using in-abstract definitions found by the reference
                          Schwartz-Hearst implementation (`abbreviations`), not scitok
Every query has a `split` of dev or test (by paper, seeded).
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT.parent / ".scitok-cache/quant-ph/abstracts.jsonl"
OUT = ROOT / "data"


def load_corpus():
    seen, docs = set(), []
    for line in open(CACHE):
        r = json.loads(line)
        yymm = r["id"].split(".")[0]
        if (
            r["id"] not in seen
            and r["categories"][:1] == ["quant-ph"]
            and len(yymm) == 4
            and yymm >= "2604"
            and "2026-04-15" <= r["datestamp"] <= "2026-10-15"
        ):
            seen.add(r["id"])
            docs.append({"id": r["id"], "title": r["title"], "abstract": r["abstract"]})
    docs.sort(key=lambda d: d["id"])
    return docs


def split_of(doc_id, dev_fraction=0.4):
    return "dev" if random.Random(doc_id).random() < dev_fraction else "test"


def acronym_swaps(docs):
    from abbreviations import schwartz_hearst

    out = []
    for d in docs:
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=d["abstract"])
        title = d["title"]
        for short, long in pairs.items():
            if len(short) < 2 or len(long.split()) < 2:
                continue
            short_re = re.compile(rf"(?<![\w-]){re.escape(short)}s?(?![\w-])")
            long_re = re.compile(re.escape(long), re.IGNORECASE)
            if short_re.search(title):
                out.append({"id": d["id"], "query": short_re.sub(long, title), "kind": "acronym->long"})
                break
            if long_re.search(title):
                out.append({"id": d["id"], "query": long_re.sub(short, title), "kind": "long->acronym"})
                break
    return out


def main():
    OUT.mkdir(exist_ok=True)
    docs = load_corpus()
    with open(OUT / "corpus.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    with open(OUT / "queries_title.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps({"qid": f"t-{d['id']}", "query": d["title"], "target": d["id"], "split": split_of(d["id"])}) + "\n")
    swaps = acronym_swaps(docs)
    with open(OUT / "queries_acronym.jsonl", "w") as f:
        for s in swaps:
            row = {"qid": f"a-{s['id']}", "query": s["query"], "target": s["id"], "kind": s["kind"], "split": split_of(s["id"])}
            f.write(json.dumps(row) + "\n")
    print(f"{len(docs)} docs, {len(docs)} title queries, {len(swaps)} acronym-swap queries", file=sys.stderr)


if __name__ == "__main__":
    main()
