"""Compare abbreviation-definition detectors on the agent-annotated quant-ph gold set.

  .venv-sci/bin/python acronym_eval.py predict-sci [dev|heldout]   # scispaCy + abbreviations
  .venv/bin/python acronym_eval.py predict-scitok [dev|heldout]    # scitok
  .venv/bin/python acronym_eval.py score [dev|heldout]

`dev` (data/acronym) was used while improving scitok's detector; `heldout`
(data/acronym/heldout) was annotated afterwards and never inspected.

Scoring is per (document, short form): strict = long form identical after
normalization (case, hyphens, whitespace); lenient = short form found with a long
form that overlaps the gold one by at least half of the gold words.
"""

import glob
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SET = sys.argv[2] if len(sys.argv) > 2 else "dev"
D = ROOT / "data" / "acronym" / ("heldout" if SET == "heldout" else "")


def docs():
    out = {}
    for f in sorted(glob.glob(str(D / "batches/*.json"))):
        for d in json.load(open(f)):
            out[d["id"]] = d["text"]
    return out


def gold():
    out = {}
    for f in sorted(glob.glob(str(D / "gold/*.json"))):
        out.update(json.load(open(f)))
    return out


def predict_sci():
    import spacy
    from abbreviations import schwartz_hearst
    from scispacy.abbreviation import AbbreviationDetector  # noqa: F401

    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe("abbreviation_detector")
    preds = {"scispacy": {}, "abbreviations": {}}
    for i, text in docs().items():
        doc = nlp(text)
        seen = {}
        for a in doc._.abbreviations:
            seen.setdefault(a.text, a._.long_form.text)
        preds["scispacy"][i] = [{"short": s, "long": l} for s, l in seen.items()]
        pairs = schwartz_hearst.extract_abbreviation_definition_pairs(doc_text=text)
        preds["abbreviations"][i] = [{"short": s, "long": l} for s, l in pairs.items()]
    json.dump(preds, open(D / "pred_sci.json", "w"), indent=1)


def predict_scitok():
    import scitok

    tok = scitok.Tokenizer(domains=[], input="plain")
    preds = {"scitok": {i: [{"short": s, "long": l} for s, l in tok.acronym_definitions(t)] for i, t in docs().items()}}
    json.dump(preds, open(D / "pred_scitok.json", "w"), indent=1)


def norm(s):
    """Lowercase, fold diacritics, drop possessives and hyphen/space differences."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"['’]s\b", "", s)
    return re.sub(r"[\s\-‐–]+", " ", s).strip()


def short_key(s):
    return re.sub(r"s$", "", s) if len(s) > 2 and s[-2:-1].isupper() else s


def score_one(pred, gold_map):
    tp_s = tp_l = n_pred = n_gold = 0
    for i, g in gold_map.items():
        gs = {short_key(x["short"]): norm(x["long"]) for x in g}
        ps = {}
        for x in pred.get(i, []):
            ps.setdefault(short_key(x["short"]), norm(x["long"]))
        n_gold += len(gs)
        n_pred += len(ps)
        for s, gl in gs.items():
            if s in ps:
                pl = ps[s]
                if pl == gl:
                    tp_s += 1
                gw, pw = set(gl.split()), set(pl.split())
                if gw and len(gw & pw) >= max(1, len(gw) / 2):
                    tp_l += 1
    out = {}
    for name, tp in (("strict", tp_s), ("lenient", tp_l)):
        p = tp / n_pred if n_pred else 0
        r = tp / n_gold if n_gold else 0
        out[name] = {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0}
    out["n_gold"], out["n_pred"] = n_gold, n_pred
    return out


def score():
    g = gold()
    preds = {}
    for f in ("pred_sci.json", "pred_scitok.json"):
        preds.update(json.load(open(D / f)))
    res = {name: score_one(p, g) for name, p in preds.items()}
    json.dump(res, open(ROOT / "results" / f"acronym_detection-{SET}.json", "w"), indent=1)
    for name, r in res.items():
        print(f"{name:14} gold={r['n_gold']} pred={r['n_pred']}  "
              + "  ".join(f"{k}: P={v['P']:.3f} R={v['R']:.3f} F1={v['F1']:.3f}" for k, v in r.items() if isinstance(v, dict)))


if __name__ == "__main__":
    {"predict-sci": predict_sci, "predict-scitok": predict_scitok, "score": score}[sys.argv[1]]()
