"""Pool top-k results of several systems for the agent-written queries and write
blind judging batches: plain text, candidates shuffled, systems not named, and
query/document ids replaced by aliases (keys stored separately).

  python pool.py write SYSTEM,SYSTEM,... [--depth 10] [--per-batch 8]
  python pool.py merge          # data/agent/judgments/*.json → data/qrels_agent.jsonl
  python pool.py coverage "SYSTEM;..." # share of top-10 results that were judged
"""

import argparse
import glob
import json
import random
import textwrap
from pathlib import Path

import metrics
import run

ROOT = Path(__file__).resolve().parent
J = ROOT / "data" / "agent"


def write(names, depth, per_batch):
    ids, texts = run.corpus("full")
    qs = [json.loads(l) for l in open(run.DATA / "queries_agent.jsonl")]
    pools = {q["qid"]: {q["source"]} for q in qs}
    for name in names:
        s = run.make(name).fit("full", texts)
        ranks = metrics.rankings(s.scores([q["query"] for q in qs]), depth)
        for q, row in zip(qs, ranks):
            pools[q["qid"]].update(ids[d] for d in row)
        print(name, "pooled", sum(map(len, pools.values())))
    text = dict(zip(ids, texts))
    out = J / "judge_v2" / "batches"
    keys = J / "judge_v2" / "keys"
    out.mkdir(parents=True, exist_ok=True)
    keys.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    rng.shuffle(qs)
    n = 0
    for b in range(0, len(qs), per_batch):
        lines, key = [], {}
        for j, q in enumerate(qs[b : b + per_batch]):
            alias_q = f"Q{b + j:03d}"
            cands = sorted(pools[q["qid"]])
            rng.shuffle(cands)
            key[alias_q] = {"qid": q["qid"], "docs": {}}
            lines += ["=" * 100, f"QUERY {alias_q}: {q['query']}", "=" * 100, ""]
            for k, d in enumerate(cands):
                alias_d = f"D{k + 1:02d}"
                key[alias_q]["docs"][alias_d] = d
                lines.append(f"[{alias_q} {alias_d}]")
                lines += textwrap.wrap(text[d], 100) + [""]
                n += 1
        name = f"batch_{b // per_batch:03d}"
        (out / f"{name}.txt").write_text("\n".join(lines))
        json.dump(key, open(keys / f"{name}.json", "w"), indent=1)
    print(f"{len(qs)} queries, {n} judgments, {(len(qs) + per_batch - 1) // per_batch} batches")


def merge():
    grades = {}
    for f in sorted(glob.glob(str(J / "judge_v2" / "judgments" / "*.json"))):
        key = json.load(open(J / "judge_v2" / "keys" / Path(f).name))
        for alias_q, g in json.load(open(f)).items():
            k = key[alias_q]
            grades[k["qid"]] = {k["docs"][d]: int(v) for d, v in g.items() if d in k["docs"]}
    with open(run.DATA / "qrels_agent.jsonl", "w") as f:
        for qid, g in sorted(grades.items()):
            f.write(json.dumps({"qid": qid, "qrels": g}) + "\n")
    print(len(grades), "queries judged")


def coverage(names, split):
    ids, texts = run.corpus("full")
    qs = run.queries("agent", split)
    for name in names:
        s = run.make(name).fit("full", texts)
        ranks = metrics.rankings(s.scores([q["query"] for q in qs]), 10)
        judged = sum(ids[d] in q["qrels"] for q, row in zip(qs, ranks) for d in row)
        print(f"{name:45} judged@10 = {judged / (10 * len(qs)):.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("systems", nargs="?", default="")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--per-batch", type=int, default=2)
    ap.add_argument("--split", default="all")
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.systems.split(";"), a.depth, a.per_batch)
    elif a.cmd == "merge":
        merge()
    else:
        coverage(a.systems.split(";"), a.split)
