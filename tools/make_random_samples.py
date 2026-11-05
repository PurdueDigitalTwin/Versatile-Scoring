"""Random control (paper Sec. 5.3, Table 4): random trajectories matched to the designed set in
the number of supervised rows and delivered to the same heads.

The designed set supervises 437,595 rung rows on navtrain (413,152 lateral + 24,443
longitudinal). This generator emits 5 random slots per scene: slots 0-3 are supervised for every
scene and slot 4 on a random 23.7% of scenes (26.2% on navtest), reproducing the counts;
`slot4_supervised` per token drives the masking in tools/make_trainsets.py, where the four dense
slots feed the DAC head and the sparse one the NC head, exactly like the lateral and longitudinal
rungs.

Free-sampled, not perturbed: each of the 8 waypoints is drawn independently and uniformly from
the per-index [p1, p99] box of the MeanFuser proposal pool, so the trajectory is an arbitrary
path through the region the planner occupies rather than a variation on a real one; headings
are recomputed from the sampled positions.

Output: exp/meanfuser_checkpoints/random_samples_<split>/samples.pkl ({token: {"proposals":
[8 + 5, 8, 3], "gt_trajectory"}}) and meta.pkl. Labelled by scripts/label_slots.sh random <split>.

Usage: python tools/make_random_samples.py --split navtrain|navtest [--limit N] [--seed S]
--limit N writes to random_samples_<split>_limitN/, never to the production directory.
"""

import argparse, pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CK  # sets env first

import numpy as np

N_SLOT = 5
# extra supervised rows needed on slot 4, as a fraction of SCENES (not rows):
#   navtrain 24,443 / 103,288 = 23.7%   navtest 3,186 / 12,146 = 26.2%
FRAC_SLOT4 = {"navtrain": 24443 / 103288, "navtest": 3186 / 12146}
MIN_STEP = 0.5                       # below this the heading is ill-conditioned; hold the previous


def sample_bank(rl):
    """Per-waypoint-index [p1, p99] box of the MF proposal pool."""
    P = np.stack([np.asarray(rl[t]["proposals"], np.float64)[:8] for t in rl])   # [T,8,8,3]
    xy = P[..., :2].reshape(-1, 8, 2)
    lo = np.percentile(xy, 1, axis=0)          # [8,2]
    hi = np.percentile(xy, 99, axis=0)
    return lo, hi


def draw(lo, hi, rng, n_slot=N_SLOT):
    """[n_slot, 8, 3] free-sampled trajectories, headings from the sampled path."""
    xy = rng.uniform(lo[None], hi[None], size=(n_slot, 8, 2))
    out = np.zeros((n_slot, 8, 3))
    out[:, :, :2] = xy
    p = np.concatenate([np.zeros((n_slot, 1, 2)), xy], 1)
    d = np.diff(p, axis=1)
    step = np.linalg.norm(d, axis=2)
    th = np.arctan2(d[:, :, 1], d[:, :, 0])
    for i in range(8):                          # hold the previous heading on tiny steps
        bad = step[:, i] < MIN_STEP
        th[bad, i] = th[bad, i - 1] if i > 0 else 0.0
    out[:, :, 2] = th
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["navtrain", "navtest"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.limit < 0:
        ap.error("--limit N needs N >= 0")
    suf = a.split
    out = CK / f"random_samples_{suf}"
    if a.limit:                                    # a partial run never lands in the production dir
        out = out.with_name(f"{out.name}_limit{a.limit}")
        print(f"--limit {a.limit}: writing to {out}")
    out.mkdir(exist_ok=True)

    rl = pickle.load(open(CK / f"plain_dataset_{suf}/dataset.pkl", "rb"))

    toks = sorted(rl)
    if a.limit: toks = toks[: a.limit]
    rng = np.random.RandomState(a.seed)
    lo, hi = sample_bank({t: rl[t] for t in toks[:5000]})   # draws nothing from rng
    print("  per-index sampling box (p1..p99) from the MF pool:")
    for i in (0, 3, 7):
        print(f"    wp{i+1}: x [{lo[i,0]:+7.2f},{hi[i,0]:+7.2f}]  y [{lo[i,1]:+6.2f},{hi[i,1]:+6.2f}]")

    bank, meta, t0 = {}, {}, time.time()
    for n, tok in enumerate(toks):
        r = draw(lo, hi, rng)
        bank[tok] = {"proposals": np.concatenate(
            [np.asarray(rl[tok]["proposals"], np.float32), r.astype(np.float32)]).astype(np.float32),
            "gt_trajectory": np.asarray(rl[tok]["gt_trajectory"], np.float32)}
        meta[tok] = {"slot4_supervised": bool(rng.rand() < FRAC_SLOT4[a.split])}
        if (n + 1) % 20000 == 0: print(f"  {n+1} scenes, {(time.time()-t0)/(n+1):.4f} s/scene", flush=True)

    s4 = sum(m["slot4_supervised"] for m in meta.values())
    pickle.dump(bank, open(out / "samples.pkl", "wb"))
    pickle.dump(meta, open(out / "meta.pkl", "wb"))
    print(f"\n  {len(bank):,} scenes in {(time.time()-t0)/60:.1f} min | slots {N_SLOT}")
    print(f"  supervised rows = {4*len(bank):,} (slots 0-3) + {s4:,} (slot 4, {s4/len(bank):.1%})"
          f" = {4*len(bank)+s4:,}")
    print(f"  target (designed set, {suf}) = {437595 if suf == 'navtrain' else 51770:,}")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
