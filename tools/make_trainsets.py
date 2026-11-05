"""Assemble the training sets (README step 4): the planner's candidate pool + the designed or
random samples + the human trajectory.

No new simulation. Every rung is anchored on the logged human trajectory, so it is
planner-independent: the same sample row and the same simulator label serve whichever
planner supplied the candidate pool. The three sample banks (step 3) and their labels cover
the same tokens, and the MeanFuser proposal rows they carry are byte-identical.

  block         samples dir           labels dir          rungs  supervision
  lateral       lateral_samples       epdms_lateral       4      dense: every scene contributes 4
  longitudinal  longitudinal_samples  epdms_longitudinal  4      per rung, meta["valid"]: ~6% of rows
                                                                 reach their target when tracked
  random        random_samples        epdms_random        5      the 5th slot on ~24% of scenes
                                                                 (meta["slot4_supervised"])

MeanFuser (--planner mf): 8 proposals, labels from components_mf_<split>.pkl
  design    0..7 proposals | 8..11 lateral rungs | 12..15 longitudinal rungs | 16 human
  random    0..7 proposals | 8..12 random        | 13 human
DiffusionDrive (--planner dd): 20 proposals, labels from components_dd_<split>.pkl, which has
no human row; the human trajectory is the same for every planner, so its label comes from the
MeanFuser labels. The rungs are the MeanFuser ones at shifted slots.
  plain     0..19 proposals | 20 human
  design    0..19 proposals | 20..23 lateral | 24..27 longitudinal | 28 human
  random    0..19 proposals | 20..24 random  | 25 human

Head targeting. Each designed rung supervises only the head of the rule it was designed
against: the lateral rungs keep their DAC label, the longitudinal rungs their NC label, and
every other component of those rows is NaN, which comp_loss drops from the other heads. A
laterally displaced trajectory would otherwise teach lane keeping, and a trajectory driven
into the lead vehicle extended comfort, from distributions those heads never see at
deployment. The random set gets the same column mask (the four dense slots to DAC, the sparse
slot to NC), so it matches the designed set in supervised rows and head assignment and differs
only in where the trajectories sit (Table 4). Unsupervised rungs (masked longitudinal rungs,
the sparse random slot, rungs absent from a simulator CSV) keep their trajectory with all-NaN
labels; the proposal and human rows are untouched, and the human row is a training target only.

Usage:
  python tools/make_trainsets.py --planner mf --split navtrain
  python tools/make_trainsets.py --planner dd --split navtest --which plain
--which defaults to all of the planner's sets. --out_root stands in for exp/ for the OUTPUT
dirs only; inputs are always read from exp/.
"""
import argparse, glob, pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CK = REPO / "exp" / "meanfuser_checkpoints"
DD = REPO / "exp" / "diffusiondrive_checkpoints"
CC = ["no_at_fault_collisions", "drivable_area_compliance", "driving_direction_compliance",
      "traffic_light_compliance", "ego_progress", "time_to_collision_within_bound",
      "lane_keeping", "history_comfort", "two_frame_extended_comfort"]
HEADS = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "TFC"]
NC_IDX, DAC_IDX = 0, 1
NPROP = {"mf": 8, "dd": 20}
RUNGS = {"lateral": ("lateral_samples", "epdms_lateral", 4),             # block -> samples dir, labels dir, n rungs
         "longitudinal": ("longitudinal_samples", "epdms_longitudinal", 4),
         "random": ("random_samples", "epdms_random", 5)}
# per planner: each set's rung blocks in slot order (a set with rungs gets the head mask)
SETS = {"mf": {"design": ["lateral", "longitudinal"], "random": ["random"]},
        "dd": {"plain": [], "design": ["lateral", "longitudinal"], "random": ["random"]}}
HELP = {"plain": "dd only: 20 proposals + human",
        "design": "proposals + 4 lateral rungs (DAC head) + 4 longitudinal rungs (NC head) + human",
        "random": "proposals + 5 random trajectories (4 dense -> DAC head, 1 sparse -> NC head) + human"}


def load_csv(dirname, k):
    cs = sorted(glob.glob(str(CK / dirname / f"k{k}_non_reactive/*.csv")))
    assert cs, f"missing CSV k{k} in {dirname}"
    df = pd.read_csv(cs[-1])
    tok = "token" if "token" in df.columns else df.columns[1]
    return df[df[tok] != "average_all_frames"].set_index(tok)


def mf_labels(split):
    """MeanFuser proposal labels and the index of the (planner-independent) human row."""
    blob = pickle.load(open(CK / f"components_mf_{split}.pkl", "rb"))
    return blob["data"], list(blob["keys"]).index("human")


def load_rungs(split, names):
    """Per block, in slot order: (name, samples, meta, {k: simulator CSV}, n rungs)."""
    rungs = []
    for name in names:
        sample_dir, csv_dir, n = RUNGS[name]
        samples = pickle.load(open(CK / f"{sample_dir}_{split}/samples.pkl", "rb"))
        meta = (pickle.load(open(CK / f"{sample_dir}_{split}/meta.pkl", "rb"))
                if name != "lateral" else None)
        sims = {k: load_csv(f"{csv_dir}_{split}", k) for k in range(8, 8 + n)}
        rungs.append((name, samples, meta, sims, n))
    return rungs


def supervised(name, meta, t, j):
    """Lateral rungs always; longitudinal rungs per meta validity; random slot 4 per scene draw."""
    if name == "longitudinal":
        return meta[t]["valid"][j]
    if name == "random":
        return j < 4 or meta[t]["slot4_supervised"]
    return True


def mf_pool(split, lab, rungs):
    """MeanFuser pool: the first sample bank fixes the token order and holds the 8 proposals
    in its slots 0..7; the human trajectory comes from the plain dataset."""
    src = pickle.load(open(CK / f"plain_dataset_{split}/dataset.pkl", "rb"))
    samples = rungs[0][1]
    toks = [t for t in samples if t in lab and t in src and all(t in r[1] for r in rungs)]
    return (toks, {t: samples[t]["proposals"][:8] for t in toks}, lab,
            {t: src[t]["gt_trajectory"] for t in toks})


def dd_pool(split, mf_lab):
    """DiffusionDrive pool: its 20 proposals and labels, tokens sorted; the human trajectory
    is read from the (human-anchored) lateral samples."""
    dd_rl = pickle.load(open(sorted((DD / f"plain_dataset_dd_{split}").glob("*_PDMS.pkl"))[-1], "rb"))
    dd_lab = pickle.load(open(DD / f"components_dd_{split}.pkl", "rb"))["data"]
    gt_src = pickle.load(open(CK / f"lateral_samples_{split}/samples.pkl", "rb"))
    toks = [t for t in sorted(dd_rl) if t in dd_lab and t in mf_lab and t in gt_src]
    drop = {"dd labels": sum(t not in dd_lab for t in dd_rl),
            "mf human label": sum(t not in mf_lab for t in dd_rl),
            "lateral samples": sum(t not in gt_src for t in dd_rl)}
    print(f"kept {len(toks)} of {len(dd_rl)} DD proposal tokens"
          + ("  (missing from: " + ", ".join(f"{k} {v}" for k, v in drop.items() if v) + ")"
             if len(toks) < len(dd_rl) else ""))
    return (toks, {t: dd_rl[t]["pred_trajectorys_array"] for t in toks}, dd_lab,
            {t: gt_src[t]["gt_trajectory"] for t in toks})


def assemble(pool, nprop, rungs, hlab, hum):
    """Candidates and labels for every token of the pool, in pool order.

    Slots 0..nprop-1 are the planner's proposals, then each rung block's rows 8..8+n of its
    samples, then the human row. Returns the dataset.pkl and components.pkl contents and the
    number of supervised rungs missing from the simulator CSVs."""
    toks, props, plab, gt = pool
    nslot = nprop + sum(r[4] for r in rungs)
    ds, olab, miss = {}, {}, 0
    for t in toks:
        C = np.full((nslot + 1, 9), np.nan, np.float32)
        E = np.full(nslot + 1, np.nan, np.float32)
        C[:nprop] = np.asarray(plab[t]["components"])[:nprop]
        E[:nprop] = np.asarray(plab[t]["epdms"])[:nprop]

        P, slot = [np.asarray(props[t], np.float32)], nprop
        for name, samples, meta, sims, n in rungs:
            P.append(np.asarray(samples[t]["proposals"], np.float32)[8:8 + n])
            for j in range(n):
                if not supervised(name, meta, t, j):
                    continue                   # stays all-NaN -> masked from every head
                if t in sims[8 + j].index:
                    r = sims[8 + j].loc[t]; C[slot + j] = [r[c] for c in CC]; E[slot + j] = r["score"]
                else: miss += 1
            slot += n

        C[nslot] = np.asarray(hlab[t]["components"])[hum]       # human row: target only
        E[nslot] = np.asarray(hlab[t]["epdms"])[hum]
        ds[t] = {"proposals": np.concatenate(P), "gt_trajectory": np.asarray(gt[t], np.float32)}
        olab[t] = {"components": C, "epdms": E}
    return ds, {"data": olab, "keys": list(range(nslot)) + ["human"], "comp_names": CC}, miss


def report(tag, cb, nprop, rungs, miss):
    olab = cb["data"]
    nslot = len(cb["keys"]) - 1
    C = np.stack([olab[t]["components"] for t in olab])
    E = np.stack([olab[t]["epdms"] for t in olab])
    print(f"{tag}: tokens={len(olab)}  slots={nslot}(+human)  missing-sim={miss}")
    print(f"  proposals 0..{nprop - 1}  EPDMS {np.nanmean(E[:, :nprop]):.4f}"
          f"   realized NC {np.nanmean(C[:, :nprop, 0]):.4f}  DAC {np.nanmean(C[:, :nprop, 1]):.4f}"
          f"  EP {np.nanmean(C[:, :nprop, 4]):.4f}  TFC {np.nanmean(C[:, :nprop, 8]):.4f}")
    slot = nprop
    for name, _, _, _, n in rungs:
        R = C[:, slot:slot + n, :]
        fin = np.isfinite(R[:, :, 0])
        print(f"  {name:12s} rungs {slot}..{slot + n - 1}  supervised rows {int(fin.sum()):,}"
              f" ({fin.mean():.1%}; {int(fin.any(1).sum()):,} scenes = {fin.any(1).mean():.1%})"
              f"   realized NC {np.nanmean(R[:, :, 0]):.4f}  DAC {np.nanmean(R[:, :, 1]):.4f}"
              f"  EP {np.nanmean(R[:, :, 4]):.4f}  TFC {np.nanmean(R[:, :, 8]):.4f}"
              f"   negatives NC=0 {int(np.nansum(R[:, :, 0] < 0.5)):,}"
              f"  DAC=0 {int(np.nansum(R[:, :, 1] < 0.5)):,}")
        slot += n
    if rungs:
        print(f"  supervised rung rows : {int(np.isfinite(C[:, nprop:nslot, 0]).sum()):,}")


def mask_heads(tag, ds, cb, nprop):
    """Head mask: rungs nprop..nprop+3 (the lateral rungs, or the dense random slots) keep only
    their DAC label, the rungs after them (the longitudinal rungs, or the sparse random slot)
    keep only their NC label; proposal and human rows untouched."""
    lab = cb["data"]
    nslot = len(cb["keys"]) - 1
    lat, lon = slice(nprop, nprop + 4), slice(nprop + 4, nslot)

    data, before, after = {}, np.zeros(9, np.int64), np.zeros(9, np.int64)
    for t in sorted(lab):
        C = np.array(lab[t]["components"], dtype=np.float32, copy=True)
        before += (~np.isnan(C[nprop:nslot])).sum(0).astype(np.int64)
        kd, kn = C[lat, DAC_IDX].copy(), C[lon, NC_IDX].copy()
        C[lat] = np.nan
        C[lon] = np.nan
        C[lat, DAC_IDX] = kd
        C[lon, NC_IDX] = kn
        after += (~np.isnan(C[nprop:nslot])).sum(0).astype(np.int64)
        d = dict(lab[t]); d["components"] = C
        data[t] = d

    print(f"{tag}: supervised rung labels per head")
    print(f"  {'head':5s} {'before':>10s} {'after':>10s}")
    for j, n in enumerate(HEADS):
        print(f"  {n:5s} {before[j]:>10,} {after[j]:>10,}")
    return ds, {**{k: v for k, v in cb.items() if k != "data"}, "data": data}


def write_set(d, ds, cb):
    d.mkdir(parents=True, exist_ok=True)
    pickle.dump(ds, open(d / "dataset.pkl", "wb"))
    pickle.dump(cb, open(d / "components.pkl", "wb"))
    print(f"-> {d}")


def main():
    ap = argparse.ArgumentParser(
        description="Assemble the training sets of one planner and split (README step 4).",
        epilog="sets:\n" + "\n".join(f"  {k:8s} {v}" for k, v in HELP.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--planner", choices=["mf", "dd"], required=True)
    ap.add_argument("--split", choices=["navtrain", "navtest"], required=True)
    ap.add_argument("--which", nargs="+", metavar="SET",
                    help="subset of the planner's sets (mf: design random; dd: plain design random); "
                         "default all")
    ap.add_argument("--out_root", type=Path, default=REPO / "exp",
                    help="stands in for exp/ for the output dirs only (default: exp/); "
                         "inputs are always read from exp/")
    a = ap.parse_args()
    nprop = NPROP[a.planner]

    sets = list(SETS[a.planner])
    unknown = set(a.which or []) - set(sets)
    if unknown:
        ap.error(f"unknown {a.planner} set(s) {sorted(unknown)}; choose from {sets}")
    which = [s for s in sets if s in (a.which or sets)]

    def out_dir(name):
        if a.planner == "mf":
            return a.out_root / "meanfuser_checkpoints" / f"versatile_dataset_{a.split}_{name}"
        return a.out_root / "diffusiondrive_checkpoints" / f"versatile_dataset_dd_{a.split}_{name}"

    hlab, hum = mf_labels(a.split)               # human row for both planners
    for name in which:
        tag = f"{a.planner}-{name} {a.split}"
        rungs = load_rungs(a.split, SETS[a.planner][name])
        pool = mf_pool(a.split, hlab, rungs) if a.planner == "mf" else dd_pool(a.split, hlab)
        ds, cb, miss = assemble(pool, nprop, rungs, hlab, hum)
        report(tag, cb, nprop, rungs, miss)
        if rungs:                                # plain has no rungs to mask
            ds, cb = mask_heads(tag, ds, cb, nprop)
        write_set(out_dir(name), ds, cb)


if __name__ == "__main__":
    main()
