"""Pre-encode both corpora with every baseline (slow models on CPU); cached in eval/cache."""

import sys
import time

import run
import systems

# Learned sparse models with a vocabulary-sized output head and the 0.5B+ models
# take over an hour per corpus on CPU, so they encode only the "full" corpus
# (agent-query task); the cheaper models also encode "abstracts" (proxy tasks).
HEAVY = ["splade-pp", "opensearch-sparse-v3", "bge-m3-sparse", "qwen3-0.6b"]
LIGHT = ["bm42", "minicoil", "bge-base", "arctic-m", "e5-base", "specter2"]
plan = [("full", n) for n in LIGHT + HEAVY] + [("abstracts", n) for n in LIGHT]
if sys.argv[1:]:
    plan = [p for p in plan if p[1] in sys.argv[1:]]
for kind, name in plan:
    _, texts = run.corpus(kind)
    if True:
        t = time.time()
        try:
            run.make(name).fit(kind, texts)
            print(f"{name} {kind}: {time.time() - t:.0f}s", flush=True)
        except Exception as e:  # keep going; report at the end
            print(f"{name} {kind}: FAILED {e!r}", flush=True)
