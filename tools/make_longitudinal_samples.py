"""Longitudinal design (paper Sec. 4): rungs that advance the ego along its own path toward the
lead vehicle, solved in executed space (Algorithm 1 with E = sim).

NAVSIM scores the motion a BatchLQRTracker + kinematic bicycle model actually drives, not the
waypoints, and tracked rungs fall short of their designed advance. So every trial advance is
pushed through the real PDMSimulator and the bumper gap is measured on the tracked motion
(executed gap < 0 predicts NC = 0 with 94% precision on 2,000 labelled rungs). A rung is emitted
only if the tracked ego reaches its target gap; otherwise it is masked (`valid` per rung, read by
tools/make_trainsets.py), so a scene that reaches -0.2 m but not -1.0 m still contributes.

On-manifold cap: the scorer sees the designed waypoints, so a rung longer than anything the
planner emits would be separable by length alone. The advance is capped at
    delta <= max_i len(proposal_i) - len(human path)
and scenes with less than MIN_ROOM of headroom are skipped.

Geometry (lead selection, margin, perturbation family) lives in tools/longitudinal_geometry.py.
Output: exp/meanfuser_checkpoints/longitudinal_samples_<split>/samples.pkl ({token: {"proposals":
[8 + 4, 8, 3], "gt_trajectory"}}) and meta.pkl (`valid`, executed gaps and advances per rung).
About 40% of scenes have an approaching lead and ~7% of all scenes yield at least one rung.
Labelled by scripts/label_slots.sh longitudinal <split>.

Usage (sys.path is pinned to navsim_epdms/, the simulator the labels use):
  python tools/make_longitudinal_samples.py --split navtrain|navtest [--limit N]
  python tools/make_longitudinal_samples.py --split navtrain --shard I N     (I = 0..N-1)
  python tools/make_longitudinal_samples.py --split navtrain --merge N
--limit N writes to longitudinal_samples_<split>_limitN/, never to the production directory.
"""

import argparse, pickle, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WS, CK, CACHE_ROOT, EPDMS_ROOT  # sets env first; imports no navsim

sys.path.insert(0, str(EPDMS_ROOT))      # must beat the editable v1 navsim install
import navsim
if EPDMS_ROOT not in Path(navsim.__file__).resolve().parents:
    sys.exit(f"navsim resolved to {navsim.__file__}, expected the v2 devkit under {EPDMS_ROOT}")

import numpy as np

import longitudinal_geometry as M
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.common.dataclasses import Trajectory

FS = TrajectorySampling(time_horizon=4, interval_length=0.1)
TS = TrajectorySampling(time_horizon=4, interval_length=0.5)
N_GRID = 25
MIN_ROOM = 0.5           # need at least this much length headroom to bother


def path_len(traj):
    p = np.concatenate([np.zeros((1, 2)), np.asarray(traj)[:, :2]])
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def exec_batch(trajs, ego0, sim):
    arrs = [get_trajectory_as_array(
        transform_trajectory(Trajectory(t.astype(np.float32), TS), ego0), FS, ego0.time_point)
        for t in trajs]
    out = sim.simulate_proposals(np.stack(arrs), ego0)
    th0 = ego0.rear_axle.heading
    R = np.array([[np.cos(th0), -np.sin(th0)], [np.sin(th0), np.cos(th0)]])
    p0 = np.array([ego0.rear_axle.x, ego0.rear_axle.y])
    xy = np.einsum("ij,btj->bti", R.T, out[:, :, :2] - p0[None, None])
    ex = np.concatenate([xy, (out[:, :, 2] - th0)[..., None]], -1)
    step = max(1, (ex.shape[1] - 1) // 8)
    idx = [min(step * (i + 1), ex.shape[1] - 1) for i in range(8)]
    return ex[:, idx, :]


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
    a = ap.parse_args()
    if a.merge is not None and a.merge < 1:
        ap.error("--merge N needs N >= 1")
    if a.limit < 0:
        ap.error("--limit N needs N >= 0")
    if a.shard and not (0 <= a.shard[0] < a.shard[1]):
        ap.error("--shard I N needs 0 <= I < N")
    suf = a.split
    out = CK / f"longitudinal_samples_{suf}"
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
        V = np.array([m["valid"] for m in meta.values()])
        print(f"merged {a.merge}: {len(bank)} tokens | scenes with >=1 valid rung "
              f"{np.mean(V.any(1)):.1%} | valid rung rows {V.mean():.1%} "
              f"({int(V.sum()):,} of {V.size:,})")
        return

    rl = pickle.load(open(CK / f"plain_dataset_{suf}/dataset.pkl", "rb"))
    logs_dir = WS / "dataset/navsim_logs" / ("trainval" if suf == "navtrain" else "test")
    t2l = pickle.load(open(CK / f"{suf}_token_to_log.pkl", "rb"))

    sim = PDMSimulator(FS)
    mcl = MetricCacheLoader(CACHE_ROOT / f"{suf}_v2_metric_cache")
    cache = set(mcl.tokens)

    toks = [t for t in sorted(rl) if t in t2l and t in cache]
    if a.shard: toks = toks[a.shard[0]::a.shard[1]]
    if a.limit: toks = toks[: a.limit]
    by = {}
    for t in toks: by.setdefault(t2l[t], []).append(t)

    bank, meta, t0, n = {}, {}, time.time(), 0
    stat = Counter()
    for lg, tks in sorted(by.items()):
        frames = pickle.load(open(logs_dir / f"{lg}.pkl", "rb"))
        idx = {f["token"]: i for i, f in enumerate(frames)}
        for tok in tks:
            i0 = idx.get(tok)
            if i0 is None or i0 + 4 >= len(frames): continue
            Mx = np.asarray(frames[i0]["ego2global"], np.float64); R0, o0 = Mx[:3, :3], Mx[:3, 3]
            gt = np.asarray(rl[tok]["gt_trajectory"], np.float64)
            props = np.asarray(rl[tok]["proposals"], np.float32)
            ags = M.agent_futures(frames, i0, R0, o0)
            tgt, g0, st, _ = M.find_lead(gt, ags) if ags else (None, np.nan, False, False)

            rungs = np.repeat(gt[None].astype(np.float32), M.N_RUNG, axis=0)
            valid = [False] * M.N_RUNG; ach = [np.nan] * M.N_RUNG; adv = [np.nan] * M.N_RUNG

            if tgt is None:
                stat["no lead"] += 1
            else:
                # on-manifold headroom: never longer than the planner's own longest proposal
                room = max(max(path_len(p) for p in props) - path_len(gt), 0.0)
                if room < MIN_ROOM:
                    stat["no length headroom"] += 1
                else:
                    grid = np.linspace(0.0, float(min(M.MAX_ADVANCE, room)), N_GRID)
                    _, s = M.path_frame(gt); w = M.ramp_weights(s[::6][1:9])
                    cands = [M.advance_traj(gt, d, w) for d in grid]
                    ego0 = mcl.get_from_token(tok).ego_state
                    ex = exec_batch(cands, ego0, sim)
                    gaps = np.array([M.gap_to_lead(ex[i], ags[tgt], st) for i in range(len(grid))])
                    got = 0
                    for j, T in enumerate(M.TARGETS):
                        hit = np.where(np.isfinite(gaps) & (gaps <= T))[0]
                        if len(hit):
                            k = int(hit[0])
                            rungs[j] = cands[k].astype(np.float32)
                            valid[j] = True; ach[j] = float(gaps[k]); adv[j] = float(grid[k]); got += 1
                    stat["feasible" if got else "lead but infeasible"] += 1

            bank[tok] = {"proposals": np.concatenate([props, rungs]).astype(np.float32),
                         "gt_trajectory": gt.astype(np.float32)}
            meta[tok] = {"log": lg, "is_lead": tgt is not None, "valid": valid,
                         "human_gap": float(g0) if tgt else np.nan,
                         "exec_gap": ach, "advance": adv, "stopped_lead": bool(st) if tgt else False}
            n += 1
            if n % 2000 == 0:
                print(f"  {n} scenes, {(time.time()-t0)/n:.3f} s/scene", flush=True)

    V = np.array([m["valid"] for m in meta.values()])
    print(f"\n{len(bank)} scenes in {(time.time()-t0)/60:.1f} min")
    for k, v in stat.most_common(): print(f"  {k:22s} {v:7d} {v/max(n,1):6.1%}")
    print(f"  scenes with >=1 valid rung : {np.mean(V.any(1)):.1%}")
    print(f"  valid rung rows            : {V.mean():.1%} ({int(V.sum()):,} of {V.size:,})")
    print(f"  of which NEGATIVE rungs    : {int(V[:, 1:].sum()):,}")
    if a.shard:
        pickle.dump({"bank": bank, "meta": meta}, open(out / f"shard_{a.shard[0]}.pkl", "wb"))
        print(f"shard {a.shard[0]} -> {out}")
    else:
        pickle.dump(bank, open(out / "samples.pkl", "wb"))
        pickle.dump(meta, open(out / "meta.pkl", "wb"))
        print(f"wrote {len(bank)} tokens -> {out}")


if __name__ == "__main__":
    main()
