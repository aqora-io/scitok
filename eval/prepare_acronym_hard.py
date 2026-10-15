"""Harder acronym task: swap an acronym/long form in the title for the alternate form
that does NOT occur in the target abstract, so the target can only be matched
through domain synonym knowledge (or the rest of the title).

Acronym pairs come from the reference Schwartz-Hearst implementation (`abbreviations`)
run over the *mining* corpus (ids 2510-2603), never over evaluation documents.
Output: data/queries_acronym_hard.jsonl
"""

import collections
import json
import re

from abbreviations import schwartz_hearst

from prepare import CACHE, OUT, split_of


def mining_docs():
    seen = set()
    for line in open(CACHE):
        r = json.loads(line)
        yymm = r["id"].split(".")[0]
        if r["id"] not in seen and r["categories"][:1] == ["quant-ph"] and len(yymm) == 4 and "2510" <= yymm <= "2603":
            seen.add(r["id"])
            yield r["abstract"]


def reference_pairs(min_count=3, dominance=0.6):
    counts = collections.defaultdict(collections.Counter)
    n = 0
    for text in mining_docs():
        n += 1
        for short, long in schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=text).items():
            if len(short) >= 2 and len(long.split()) >= 2:
                counts[short][long.lower()] += 1
    pairs = {}
    for short, longs in counts.items():
        (long, c), total = longs.most_common(1)[0], sum(longs.values())
        if c >= min_count and c / total >= dominance:
            pairs[short] = long
    return pairs, n


def main():
    pairs, n = reference_pairs()
    docs = [json.loads(l) for l in open(OUT / "corpus.jsonl")]
    rows = []
    for d in docs:
        title, abstract = d["title"], d["abstract"]
        for short, long in sorted(pairs.items(), key=lambda p: -len(p[1])):
            short_re = re.compile(rf"(?<![\w-]){re.escape(short)}s?(?![\w-])")
            long_re = re.compile(re.escape(long).replace(r"\ ", r"[\s-]+").replace(r"\-", r"[\s-]+"), re.IGNORECASE)
            if short_re.search(title) and not long_re.search(abstract):
                rows.append({"qid": f"h-{d['id']}", "query": short_re.sub(long, title), "target": d["id"], "kind": "acronym->long", "pair": [short, long]})
                break
            if long_re.search(title) and not short_re.search(abstract):
                rows.append({"qid": f"h-{d['id']}", "query": long_re.sub(short, title), "target": d["id"], "kind": "long->acronym", "pair": [short, long]})
                break
    with open(OUT / "queries_acronym_hard.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(dict(r, split=split_of(r["target"]))) + "\n")
    kinds = collections.Counter(r["kind"] for r in rows)
    print(f"{len(pairs)} reference pairs from {n} mining abstracts; {len(rows)} queries {dict(kinds)}")


if __name__ == "__main__":
    main()
