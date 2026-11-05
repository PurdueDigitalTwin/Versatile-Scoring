"""token -> log_name map for one split, read off the EPDMS metric-cache metadata.

The cache stores every scene at  {cache}/{log_name}/unknown/{token}/metric_cache.pkl
and lists those paths in metadata/*_metadata_node_*.csv, so the map falls straight out
of the paths -- no log pickles are opened.

Consumers: the sample generators (tools/make_lateral_samples.py and
tools/make_longitudinal_samples.py open the log pickle a token belongs to) and
run_scorer_training.py --logs (log-level
validation hold-out; the split is a deterministic function of the set of log names).

Usage:  python tools/make_token_to_log.py --split navtrain
        python tools/make_token_to_log.py --split navtest
Output: exp/meanfuser_checkpoints/{split}_token_to_log.pkl      {token: log_name}
"""
import argparse, glob, os, pickle
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("NAVSIM_WORKSPACE") or REPO.parent)   # same default as scripts/env.sh
CK = REPO / "exp" / "meanfuser_checkpoints"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["navtrain", "navtest"], required=True)
    a = ap.parse_args()

    md = WS / "cache" / f"{a.split}_v2_metric_cache" / "metadata"
    meta = sorted(glob.glob(str(md / "*_metadata_node_*.csv")))
    assert meta, (f"no metadata CSV under {md}; build it first with "
                  f"bash scripts/build_cache.sh v2_metric {a.split}")

    t2l = {}
    for m in meta:
        for p in pd.read_csv(m)["file_name"]:
            parts = Path(p).parts                     # .../{log}/unknown/{token}/metric_cache.pkl
            token, log = parts[-2], parts[-4]
            assert t2l.get(token, log) == log, f"token {token} in two logs"
            t2l[token] = log

    out = CK / f"{a.split}_token_to_log.pkl"
    pickle.dump(t2l, open(out, "wb"))
    print(f"{a.split}: {len(t2l)} tokens in {len(set(t2l.values()))} logs -> {out}")


if __name__ == "__main__":
    main()
