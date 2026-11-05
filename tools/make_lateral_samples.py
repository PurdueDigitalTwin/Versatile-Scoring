"""Lateral design (paper Sec. 4): rungs that walk the human path toward the drivable-area boundary.

Every scene gets K = 4 rungs anchored on the logged human trajectory. The human's own
margin c is the signed distance from its swept footprint to the drivable-area boundary
(all four corners at all eight waypoints, + = inside). The rungs are lateral offsets of
the human path along the unit normal toward the nearest boundary, ramped by the
smoothstep w(s) of Eq. 4 so the excursion develops like a lane change, with headings
recomputed from the offset path. Their target margins are set relative to c (Eq. 7):

    m1  +min(0.30 c, 0.80)              near-miss, still inside
    m2  -clip(0.10 c, 0.15, 0.40)       graded violations
    m3  -clip(0.25 c, 0.35, 0.80)
    m4  -clip(0.50 c, 0.60, 1.40)

The floors keep a target far enough out to survive the tracker; the caps keep it close to
the boundary (the human's clearance is bimodal: 35% of scenes below 1 m, 54% above 3 m).
The smallest offset reaching each target is solved by a grid search plus bisection on the
footprint margin (Eq. 8); a target outside the kinematic budget takes the closest offset,
so no scene is discarded. m2 is deliberately marginal: the labels come from the
simulator, so a rung the tracker pulls back inside returns DAC = 1 and is a hard positive
on the boundary.

Output: exp/meanfuser_checkpoints/lateral_samples_<split>/samples.pkl, {token: {"proposals":
[8 + 4, 8, 3] (the 8 MeanFuser proposals, then the rungs), "gt_trajectory": [8, 3]}}, and
meta.pkl with the clearance, targets and achieved margins per scene. The rungs are
labelled by scripts/label_slots.sh lateral <split> and assembled by tools/make_trainsets.py.

Usage: python tools/make_lateral_samples.py --split navtrain|navtest [--limit N] [--dry]
       python tools/make_lateral_samples.py --split navtrain --shard I N     (I = 0..N-1)
       python tools/make_lateral_samples.py --split navtrain --merge N
--limit N writes to lateral_samples_<split>_limitN/, never to the production directory.
"""

import argparse, pickle, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import densify, drivable_union, map_api_for, WS, CK, EGO_L, EGO_W  # sets env first

import numpy as np
import shapely
from shapely.geometry import Point

REAR_AXLE_TO_CENTER = 1.461          # nuplan pacifica: pose is the REAR AXLE


# ------------------------------------------------- footprint geometry ---

def footprint_corners(traj, R, o):
    """[8,3] ego-frame trajectory -> [32,2] global footprint corner points."""
    x, y, th = traj[:, 0], traj[:, 1], traj[:, 2]
    cx, cy = x + REAR_AXLE_TO_CENTER * np.cos(th), y + REAR_AXLE_TO_CENTER * np.sin(th)
    hl, hw = EGO_L / 2.0, EGO_W / 2.0
    loc = np.array([[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]])          # [4,2]
    c, s = np.cos(th)[:, None], np.sin(th)[:, None]
    px = cx[:, None] + loc[:, 0] * c - loc[:, 1] * s                       # [8,4]
    py = cy[:, None] + loc[:, 0] * s + loc[:, 1] * c
    ego = np.stack([px.ravel(), py.ravel()], -1)                           # [32,2]
    return (R[:2, :2] @ ego.T).T + o[:2]                                   # -> global


def path_margin(traj, R, o, dr, boundary):
    """Signed min margin of the swept footprint to the drivable boundary.
    + = fully inside by that much, - = worst corner is outside by that much."""
    g = footprint_corners(traj, R, o)
    pts = shapely.points(g)
    d = shapely.distance(pts, boundary)
    inside = shapely.contains_xy(dr, g[:, 0], g[:, 1])
    return float(np.min(np.where(inside, d, -d)))


def closest_normal(traj, R, o, dr):
    """Ego-frame unit normal pointing at the nearest boundary, + arc of closest
    approach (centreline is enough to fix the DIRECTION; the margin itself is
    measured on the footprint)."""
    pts, sa = densify(traj.astype(np.float64))
    g = (R[:2, :2] @ pts.T).T + o[:2]
    bd = dr.boundary
    sub, ssub = g[::3], sa[::3]
    d = shapely.distance(shapely.points(sub), bd)
    inside = shapely.contains_xy(dr, sub[:, 0], sub[:, 1])
    k = int(np.argmin(np.where(inside, d, -d)))
    bp = bd.interpolate(bd.project(Point(*sub[k])))
    nrm = R[:2, :2].T @ np.array([bp.x - sub[k][0], bp.y - sub[k][1]])
    n = np.linalg.norm(nrm)
    return (nrm / n if n > 1e-3 else np.array([0.0, 1.0])), float(ssub[k]), float(sa[-1])

N_RUNG = 4
S_RAMP = 10.0      # lateral excursion develops over this arc length
DELTA_FRAC = 0.15  # |offset| <= DELTA_FRAC * path length (kinematic budget)
MIN_STEP = 0.5     # heading is only recomputed where the step is this long
TOL = 0.05
LABELS = ("R1 +min(.30c,.8)", "R2 -clip(.10c,.15,.40)",
          "R3 -clip(.25c,.35,.80)", "R4 -clip(.50c,.60,1.40)")


def ramp_weights(wp_arc, s_ramp=S_RAMP):
    """Monotone smoothstep ramp (Eq. 4): 0 at the ego pose, 1 by s_ramp, held after.

    A ramp anchored at w(0) = 0 is the only profile a vehicle can actually execute: the
    rung bends into a lane change. A bump peaking at the point of closest approach, or a
    rigid constant offset, displaces waypoint 1 and produced an instantaneous swerve
    (> 0.4 rad of heading change per 0.5 s step) in 58% / 74% of rungs."""
    x = np.clip(wp_arc / s_ramp, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def offset_traj_guarded(base, delta, w, normal, min_step=MIN_STEP):
    """Lateral offset + heading recomputation guarded against ill-conditioning.

    For a creeping ego (22% of scenes travel < 10 m in the 4 s horizon)
    consecutive waypoints sit ~0.3 m apart, so atan2 on their difference swings
    wildly under any perturbation -- the source of 6+ rad outliers. Where the
    step is shorter than `min_step` the base heading is kept instead."""
    out = base.copy().astype(np.float64)
    out[:, 0] += delta * w * normal[0]
    out[:, 1] += delta * w * normal[1]
    p = np.concatenate([np.zeros((1, 2)), out[:, :2]], 0)
    d = np.diff(p, axis=0)
    step = np.linalg.norm(d, axis=1)
    out[:, 2] = np.where(step >= min_step, np.arctan2(d[:, 1], d[:, 0]), base[:, 2])
    return out


def targets_for(c):
    """Human clearance c (m) -> the 4 rung target margins (m)."""
    c = max(float(c), 0.05)
    return (min(0.30 * c, 0.80),
            -float(np.clip(0.10 * c, 0.15, 0.40)),
            -float(np.clip(0.25 * c, 0.35, 0.80)),
            -float(np.clip(0.50 * c, 0.60, 1.40)))


def build_scene(gt, R, o, dr, boundary):
    """-> (rungs [4,8,3], targets, achieved, c). Never discards; never NaN."""
    c = path_margin(gt, R, o, dr, boundary)
    tg = targets_for(c)
    nrm, _, _ = closest_normal(gt, R, o, dr)
    _, wp_s = densify(gt); wp_arc = wp_s[::6][1:9]; L = float(wp_s[-1])
    w = ramp_weights(wp_arc)
    lim = max(DELTA_FRAC * L, 0.2)                 # kinematic budget on |offset|
    ds = np.arange(-lim, lim + 1e-9, max(lim / 30.0, 0.02))
    ms = np.array([path_margin(offset_traj_guarded(gt, d, w, nrm), R, o, dr, boundary) for d in ds])
    rungs, ach = [], []
    for T in tg:
        g = ms - T
        idx = np.where(np.sign(g[:-1]) * np.sign(g[1:]) <= 0)[0]
        if len(idx):                               # bisect the cell nearest delta=0
            i = idx[np.argmin(np.abs(0.5 * (ds[idx] + ds[idx + 1])))]
            lo, hi, flo = ds[i], ds[i + 1], g[i]
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                fm = path_margin(offset_traj_guarded(gt, mid, w, nrm), R, o, dr, boundary) - T
                if abs(fm) < TOL: break
                if np.sign(fm) == np.sign(flo): lo, flo = mid, fm
                else: hi = mid
            dl = 0.5 * (lo + hi)
        else:                                      # unreachable inside the budget:
            dl = float(ds[int(np.argmin(np.abs(ms - T)))])   # closest we can get
        rung = offset_traj_guarded(gt, dl, w, nrm)
        rungs.append(rung.astype(np.float32))
        ach.append(path_margin(rung, R, o, dr, boundary))
    return np.stack(rungs), tg, ach, c


def check_shards(out, n):
    """--merge N must find exactly the shard_0..shard_{N-1}.pkl that --shard I N writes."""
    want = {f"shard_{i}.pkl" for i in range(n)}
    have = {p.name for p in out.glob("shard_*.pkl")}
    missing, extra = sorted(want - have), sorted(have - want)
    if missing or extra:
        print(f"--merge {n} in {out}: missing {missing or 'none'}; unexpected (stale?) "
              f"{extra or 'none'}. Re-run the missing shards or remove the stale ones.",
              file=sys.stderr)
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["navtrain", "navtest"], required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", nargs=2, type=int)
    ap.add_argument("--merge", type=int)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.merge is not None and a.merge < 1:
        ap.error("--merge N needs N >= 1")
    if a.limit < 0:
        ap.error("--limit N needs N >= 0")
    if a.shard and not (0 <= a.shard[0] < a.shard[1]):
        ap.error("--shard I N needs 0 <= I < N")

    rl = pickle.load(open(CK / f"plain_dataset_{a.split}/dataset.pkl", "rb"))
    logs_dir = WS / "dataset/navsim_logs" / ("trainval" if a.split == "navtrain" else "test")
    out = CK / f"lateral_samples_{a.split}"
    log_src = CK / f"{a.split}_token_to_log.pkl"
    if a.limit:                                    # a partial run never lands in the production dir
        out = out.with_name(f"{out.name}_limit{a.limit}")
        print(f"--limit {a.limit}: writing to {out}")
    out.mkdir(exist_ok=True)

    if a.merge:
        check_shards(out, a.merge)
        bank, meta = {}, {}
        for i in range(a.merge):
            sh = pickle.load(open(out / f"shard_{i}.pkl", "rb"))
            bank.update(sh["bank"]); meta.update(sh["meta"])
        pickle.dump(bank, open(out / "samples.pkl", "wb"))
        pickle.dump(meta, open(out / "meta.pkl", "wb"))
        print(f"merged {a.merge}: {len(bank)} tokens -> {out}"); return

    log_of = pickle.load(open(log_src, "rb"))      # flat {token: log_name}
    toks = [t for t in sorted(rl) if t in log_of]
    if a.shard: toks = toks[a.shard[0]::a.shard[1]]
    if a.limit: toks = toks[: a.limit]
    by = {}
    for t in toks: by.setdefault(log_of[t], []).append(t)

    bank, meta, err, C = {}, {}, [], []
    for lg, tk in sorted(by.items()):
        frames = pickle.load(open(logs_dir / f"{lg}.pkl", "rb"))
        fmap = {f["token"]: f for f in frames}; mapi = map_api_for(frames[0]["map_location"])
        for tok in tk:
            f = fmap.get(tok)
            if f is None: continue
            M = np.asarray(f["ego2global"], np.float64); R, o = M[:3, :3], M[:3, 3]
            dr = drivable_union(mapi, o[0], o[1])
            if dr is None or dr.is_empty: continue
            gt = np.asarray(rl[tok]["gt_trajectory"], np.float64)
            rungs, tg, ach, c = build_scene(gt, R, o, dr, dr.boundary)
            bank[tok] = {"proposals": np.concatenate(
                [np.asarray(rl[tok]["proposals"], np.float32), rungs]).astype(np.float32),
                "gt_trajectory": gt.astype(np.float32)}
            meta[tok] = {"c": c, "targets": list(tg), "achieved": ach, "log": lg}
            err += [abs(g - t) for g, t in zip(ach, tg)]; C.append(c)
    err = np.array(err); C = np.array(C)
    print(f"scenes {len(bank)}  rungs {len(err)}  ({len(err)/max(len(bank),1):.2f}/scene, no discard)")
    print(f"human clearance c: p25 {np.percentile(C,25):.2f}  median {np.median(C):.2f}  p75 {np.percentile(C,75):.2f}")
    print(f"target error: median {np.median(err):.3f} m  within 5cm {np.mean(err<=0.05):.1%}  "
          f"within 20cm {np.mean(err<=0.20):.1%}")
    if a.dry: print("--dry: nothing written"); return
    if a.shard:
        pickle.dump({"bank": bank, "meta": meta}, open(out / f"shard_{a.shard[0]}.pkl", "wb"))
        print(f"shard {a.shard[0]} -> {out}")
    else:
        pickle.dump(bank, open(out / "samples.pkl", "wb"))
        pickle.dump(meta, open(out / "meta.pkl", "wb"))
        print(f"wrote {len(bank)} tokens -> {out}")


if __name__ == "__main__":
    main()
