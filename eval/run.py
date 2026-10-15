"""Evaluate systems on the quant-ph tasks.

  python run.py --tasks title,acronym --split dev --systems "scitok;bm25s;fastembed-bm25"

Tasks: title (known-item, title → abstract), acronym (known-item with swapped
acronyms), agent (agent-written queries with judged pools; needs data/qrels_agent.jsonl).
Results are merged into results/<task>-<split>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import metrics
import systems

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


def load_jsonl(path):
    return [json.loads(line) for line in open(path)]


def corpus(kind):
    docs = load_jsonl(DATA / "corpus.jsonl")
    texts = [d["abstract"] if kind == "abstracts" else f"{d['title']}.\n{d['abstract']}" for d in docs]
    return [d["id"] for d in docs], texts


TASKS = {
    # task: (corpus kind, query file)
    "title": ("abstracts", "queries_title.jsonl"),
    "acronym": ("abstracts", "queries_acronym.jsonl"),
    "acronym_hard": ("abstracts", "queries_acronym_hard.jsonl"),
    "agent": ("full", "queries_agent.jsonl"),
}


def queries(task, split):
    qs = [q for q in load_jsonl(DATA / TASKS[task][1]) if split == "all" or q["split"] == split]
    if task == "agent":
        qrels = {r["qid"]: r["qrels"] for r in load_jsonl(DATA / "qrels_agent.jsonl")}
        qs = [dict(q, qrels=qrels[q["qid"]]) for q in qs if q["qid"] in qrels]
    return qs


def make(name):
    if name.startswith("rrf:"):
        a, b = name[4:].split("+")
        return systems.Hybrid(make(a), make(b))
    if name.startswith("scitok"):
        # scitok or scitok:k1=0.9,b=0.6,concept_weight=2
        # domain=<variant> selects eval/domains/variants/<variant> (or "none")
        opts = {}
        if ":" in name:
            for kv in name.split(":", 1)[1].split(","):
                k, v = kv.split("=")
                if k == "domain":
                    opts["domain_paths"] = () if v == "none" else (str(systems.ROOT / "domains" / "variants" / v),)
                elif v in ("true", "false"):
                    opts[k] = v == "true"
                else:
                    opts[k] = float(v)
        return systems.Scitok(**opts)
    return systems.baselines()[name]()


def evaluate(system_names, tasks, split, save=True):
    out = {}
    fitted = {}
    for task in tasks:
        kind = TASKS[task][0]
        ids, texts = corpus(kind)
        index = {d: i for i, d in enumerate(ids)}
        qs = queries(task, split)
        res = {}
        for name in system_names:
            t0 = time.time()
            key = (name, kind)
            if key not in fitted:
                fitted[key] = _fit(make(name), kind, texts)
            sc = fitted[key].scores([q["query"] for q in qs])
            if task == "agent":
                m = metrics.graded(sc, [q["qrels"] for q in qs], ids)
            else:
                m = metrics.known_item(sc, [q["target"] for q in qs], index)
            m["n"] = len(qs)
            res[name] = m
            print(f"{task:8} {split:5} {name:45} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k != "n") + f"  ({time.time()-t0:.0f}s)", file=sys.stderr)
        out[task] = res
        if save:
            RESULTS.mkdir(exist_ok=True)
            path = RESULTS / f"{task}-{split}.json"
            prev = json.loads(path.read_text()) if path.exists() else {}
            prev.update(res)
            path.write_text(json.dumps(prev, indent=1, sort_keys=True))
    return out


def _fit(system, kind, texts):
    return system.fit(kind, texts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="title,acronym")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--systems", default="scitok;whitespace-bm25;bm25s;fastembed-bm25")
    a = ap.parse_args()
    evaluate(a.systems.split(";"), a.tasks.split(","), a.split)
