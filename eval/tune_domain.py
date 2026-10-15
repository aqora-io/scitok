"""Grid over mining parameters: build eval domains from the mining corpus only
(2025-10-15..2026-04-14) and score them on dev tasks.

  python tune_domain.py [tasks]   # default: title,acronym_hard
"""

import itertools
import shutil
import subprocess
import sys
from pathlib import Path

import run
import systems

ROOT = Path(__file__).resolve().parent
LEARN = ROOT.parent / "target/release/scitok-learn"
VARIANTS = ROOT / "domains" / "variants"
CURATED = ROOT / "domains" / "quant-ph" / "curated.toml"


def build(name, args, curated=True):
    out = VARIANTS / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    if curated:
        shutil.copy(CURATED, out / "curated.toml")
    cmd = [str(LEARN), "build", "--category", "quant-ph", "--from", "2025-10-15", "--until", "2026-04-14",
           "--cache-dir", str(ROOT.parent / ".scitok-cache"), "--out", str(out), *args]
    subprocess.run(cmd, check=True, capture_output=True)
    return out


def main():
    tasks = (sys.argv[1] if len(sys.argv) > 1 else "title,acronym_hard").split(",")
    grid = {"none": None, "curated-only": "curated"}
    for mc, npmi, mp in itertools.product((3, 5, 10), (0.4, 0.5, 0.6), (500, 1500, 3000)):
        grid[f"mc{mc}-npmi{npmi}-mp{mp}"] = ["--min-count", str(mc), "--min-npmi", str(npmi), "--max-phrases", str(mp)]
    grid["mined-only"] = ("nocurated", ["--min-count", "5"])
    for name, spec in grid.items():
        if spec is None:
            paths = ()
        elif spec == "curated":
            d = VARIANTS / name
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy(CURATED, d / "curated.toml")
            paths = (str(d),)
        elif isinstance(spec, tuple):
            paths = (str(build(name, spec[1], curated=False)),)
        else:
            paths = (str(build(name, spec)),)
        for opts in ({"compounds": False},):
            orig = run.make
            run.make = lambda n, p=paths, o=opts: systems.Scitok(domain_paths=p, **o)
            res = run.evaluate([f"domain={name}"], tasks, "dev", save=False)
            run.make = orig


if __name__ == "__main__":
    main()
