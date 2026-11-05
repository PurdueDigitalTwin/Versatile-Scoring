"""Dump the per-proposal EPDMS COMPONENT tensor from the deploy-index CSVs.

Each deploy run (k0..kN, gt) scored one stored trajectory per scene; its CSV has
the 9 EPDMS sub-metric columns per token. We stack them into {token: [N, 9]} and
also keep the scalar EPDMS `score` per proposal + the PDMS reward (from the
dataset) for the probe.

Self-check: recompose EPDMS from the 9 components via the scorer's exact formula
  EPDMS = (NC*DAC*DDC*TLC) * (5*EP + 5*TTC + 2*LK + 2*HC + 2*TFC) / 16
and compare to the CSV `score`. If max|err| is tiny, the components + formula are
consistent and safe to train on; otherwise nothing is written and the exit code is 1.

Only tokens present in the dataset and in every deploy CSV are kept; the count of
kept tokens and the tokens each CSV is missing are printed.

Usage:
  python tools/dump_components.py --deploy_dir exp/meanfuser_checkpoints/epdms_proposals_navtest \
      --n 8 --include_arm --include_human \
      --dataset exp/meanfuser_checkpoints/plain_dataset_navtest/dataset.pkl \
      --out exp/meanfuser_checkpoints/components_mf_navtest.pkl
  python tools/dump_components.py --deploy_dir exp/diffusiondrive_checkpoints/epdms_proposals_navtest \
      --n 20 --include_arm \
      --dataset exp/diffusiondrive_checkpoints/plain_dataset_dd_navtest/dataset.pkl \
      --out exp/diffusiondrive_checkpoints/components_dd_navtest.pkl
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

COMP = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
        "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
        "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
GATES = slice(0, 4)        # NC, DAC, DDC, TLC  (multiplicative)
WEIGHTS = np.array([5., 5., 2., 2., 2.])   # EP, TTC, LK, HC, TFC


def compose_epdms(C):  # C: [..., 9] -> [...]
    # NaN-robust: weighted metrics that are NaN (e.g. two-frame comfort on the
    # first frame of a log) are masked out and the weights renormalized -- this
    # matches the scorer, which excludes unavailable metrics from the average.
    gate = C[..., 0] * C[..., 1] * C[..., 2] * C[..., 3]
    vals = C[..., 4:9]                       # EP, TTC, LK, HC, TFC
    present = ~np.isnan(vals)
    num = np.where(present, vals, 0.0) * WEIGHTS          # [...,5]
    den = (present * WEIGHTS).sum(-1)
    return gate * (num.sum(-1) / den)


# the deploy sweep names the human run after the agent's index string ("gt"),
# while the labels pkl exposes it under the clearer name "human".
DIR_ALIAS = {"human": "gt"}


def load_one(deploy_dir, key):
    sub = DIR_ALIAS.get(key, key)
    csvs = sorted(glob.glob(os.path.join(deploy_dir, f"k{sub}_*", "*.csv")))
    assert csvs, f"no CSV for k{sub} ({deploy_dir})"
    df = pd.read_csv(csvs[-1])
    tok = "token" if "token" in df.columns else df.columns[1]
    df = df[df[tok] != "average_all_frames"].set_index(tok)
    return df[COMP].astype(float), df["score"].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy_dir", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--include_arm", action="store_true",
                    help="append the ARM column. EVAL-ONLY: the scorer never trains on it; "
                         "it is the planner's-own-choice baseline (the `arm` slot check of the README).")
    ap.add_argument("--include_human", action="store_true",
                    help="append the HUMAN (logged GT) row from kgt_*; this is the ONLY extra "
                         "training target -- run_scorer_training.py --human consumes it")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keys = list(range(args.n))
    if args.include_arm:
        keys.append("arm")
    if args.include_human:
        keys.append("human")
    comp_d, score_d = {}, {}
    for k in keys:
        comp_d[k], score_d[k] = load_one(args.deploy_dir, k)

    rl = pickle.load(open(args.dataset, "rb"))
    common = set(rl)
    for k in keys:
        common &= set(comp_d[k].index)
    tokens = sorted(common)
    print(f"kept {len(tokens)} of {len(rl)} dataset tokens  candidates: {len(keys)}  (keys={keys})")
    if len(tokens) < len(rl):
        print(f"WARNING: {len(rl) - len(tokens)} dataset tokens dropped; missing from the CSV of")
        for k in keys:
            n_miss = len(set(rl) - set(comp_d[k].index))
            if n_miss:
                print(f"  k{DIR_ALIAS.get(k, k)}: {n_miss} tokens")
    if not tokens:
        sys.exit(f"no token is in both {args.dataset} and every CSV: {args.out} not written")

    out = {}
    all_err = []
    for t in tokens:
        C = np.stack([comp_d[k].loc[t].values for k in keys]).astype(np.float64)  # [N,9]
        s = np.array([score_d[k].loc[t] for k in keys])                            # [N]
        extra_pdms = ([rl[t]["arm_reward"]] if args.include_arm else []) \
                     + ([rl[t]["gt_reward"]] if args.include_human else [])
        pdms = np.array(list(rl[t]["proposal_scores"]) + extra_pdms)
        all_err.append(np.abs(compose_epdms(C) - s))
        out[t] = {"components": C.astype(np.float32), "epdms": s.astype(np.float32),
                  "pdms": pdms.astype(np.float32)}
    all_err = np.concatenate(all_err)
    ok = all_err.max() < 1e-3
    print(f"composition self-check: max|err|={all_err.max():.2e}  mean|err|={all_err.mean():.2e}  "
          f"{'PASS' if ok else '*** FORMULA MISMATCH ***'}")
    if not ok:
        sys.exit(f"self-check failed: {args.out} not written")

    pickle.dump({"data": out, "comp_names": COMP, "tokens": tokens, "keys": keys}, open(args.out, "wb"))
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
