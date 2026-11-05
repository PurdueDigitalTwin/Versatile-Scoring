"""Longitudinal design (paper Sec. 4), geometry: the lead vehicle, the bumper-gap margin and the
longitudinal perturbation family. tools/make_longitudinal_samples.py imports this module and
solves the rungs with the simulator in the loop.

The ego advances along its own path (Eq. 6): waypoint i moves to arc length s_i + delta w(s_i),
with the smoothstep ramp w of Eq. 4 shared with the lateral design. The path itself is untouched,
so DAC, LK and DDC are unchanged by construction while the rung stays on the logged path; past
its end the ego continues along its final heading.

Margin (Sec. 4.1): M_lon is the minimum gap over the horizon from the ego's front-bumper line to
the chosen lead vehicle, counting only steps where that vehicle is ahead. Rungs are emitted only
for scenes with a genuine lead-follow conflict: a vehicle ahead of the ego along its own path,
inside a lane-width corridor around it (CORRIDOR_W / 2 either side) and being approached (the
gap shrinks over the horizon). Other scenes emit nothing; the scene filter is the point.

Fault model (NAVSIM's get_collision_type): rear-ending a moving lead with the front bumper is
ACTIVE_FRONT and any contact with a stopped lead is STOPPED_TRACK, both at fault, which is what
a longitudinal overshoot produces.

Targets (Sec. 4.4), metres of bumper gap, + = clear, - = overlap:
    +0.5  near-miss (still NC = 1)      -0.2  -0.6  -1.0  graded contacts
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import densify, EGO_L, EGO_W  # sets env first
from make_lateral_samples import REAR_AXLE_TO_CENTER, ramp_weights

import numpy as np
from shapely.geometry import Polygon, LineString

TARGETS = (0.5, -0.2, -0.6, -1.0)     # 1 positive + 3 negatives
N_RUNG = len(TARGETS)
HORIZON, DT = 8, 0.5
STOP_SPEED = 0.05                     # NAVSIM's stopped threshold, m/s
CORRIDOR_W = 3.5                      # lane-width corridor for "is the lead"
MAX_ADVANCE = 16.0                    # cap on extra distance travelled, m
# Chosen by sweep (250 navtest lead scenes): at-fault yield 60.1 / 68.5 / 70.5 / 72.1% at caps
# 12 / 16 / 20 / 26 m, but the p90 of the implied acceleration climbs 4.5 -> 5.5 -> 6.7 -> 8.7
# m/s^2; past ~16 m the extra faults need accelerations no Pacifica can produce.
AT_FAULT = {"STOPPED_TRACK", "ACTIVE_FRONT"}


def _rect(cx, cy, L, W, yaw):
    """Corners ordered FL, RL, RR, FR to match nuplan's OrientedBox.all_corners."""
    loc = np.array([[L/2, W/2], [-L/2, W/2], [-L/2, -W/2], [L/2, -W/2]])
    c, s = np.cos(yaw), np.sin(yaw)
    return Polygon(np.stack([cx + loc[:, 0]*c - loc[:, 1]*s,
                             cy + loc[:, 0]*s + loc[:, 1]*c], -1))


def ego_polys(traj):
    return [_rect(x + REAR_AXLE_TO_CENTER*np.cos(th), y + REAR_AXLE_TO_CENTER*np.sin(th),
                  EGO_L, EGO_W, th) for x, y, th in traj]


def front_line(poly):
    """The scorer's front-bumper edge: exterior coords[0] -> coords[3]."""
    c = poly.exterior.coords
    return LineString([c[0], c[3]])


def agent_futures(frames, i0, R0, o0, horizon=HORIZON):
    """{track_token: {step: [x, y, z, L, W, H, yaw]}} in the ego frame at i0."""
    out = {}
    yaw0 = np.arctan2(R0[1, 0], R0[0, 0])
    for k in range(1, horizon + 1):
        if i0 + k >= len(frames):
            break
        fk = frames[i0 + k]
        Mk = np.asarray(fk["ego2global"], np.float64); Rk, ok = Mk[:3, :3], Mk[:3, 3]
        yawk = np.arctan2(Rk[1, 0], Rk[0, 0])
        a = fk["anns"]
        for b, nm, tk in zip(np.asarray(a["gt_boxes"], np.float64), a["gt_names"], a["track_tokens"]):
            if nm != "vehicle":
                continue
            g = (Rk[:2, :2] @ b[:2]) + ok[:2]
            loc = R0[:2, :2].T @ (g - o0[:2])
            out.setdefault(tk, {})[k] = np.array(
                [loc[0], loc[1], b[2], b[3], b[4], b[5], b[6] + yawk - yaw0])
    return {tk: v for tk, v in out.items() if len(v) >= HORIZON // 2}


def agent_speed(fut, k):
    if k in fut and k + 1 in fut: return float(np.hypot(*(fut[k+1][:2] - fut[k][:2])) / DT)
    if k in fut and k - 1 in fut: return float(np.hypot(*(fut[k][:2] - fut[k-1][:2])) / DT)
    return np.nan


def is_stopped_track(fut):
    sp = [agent_speed(fut, k) for k in sorted(fut)]
    sp = [s for s in sp if np.isfinite(s)]
    return bool(sp) and float(np.median(sp)) <= STOP_SPEED


# --------------------------------------------------------- lead selection ---

def path_frame(gt):
    """Dense path points + arc length, with the ego pose prepended as s=0."""
    pts, s = densify(gt.astype(np.float64))
    return pts, s


def station_offset(pts, s, xy):
    """Project a point onto the ego path -> (arc length along, signed lateral)."""
    d = pts - np.asarray(xy)[None]
    k = int(np.argmin(np.einsum("ij,ij->i", d, d)))
    tan = (pts[min(k+1, len(pts)-1)] - pts[max(k-1, 0)])
    n = np.linalg.norm(tan)
    if n < 1e-6:
        return float(s[k]), float(np.linalg.norm(d[k]))
    tan = tan / n
    nrm = np.array([-tan[1], tan[0]])
    return float(s[k]), float(np.dot(xy - pts[k], nrm))


def find_lead(gt, agents):
    """The lead vehicle the ego is approaching, or None.

    Returns (track_token, gap0, stopped, closing) where gap0 is the front-bumper
    gap at the step of closest approach on the ORIGINAL trajectory."""
    pts, s = path_frame(gt)
    best = None
    for tk, fut in agents.items():
        ks = sorted(fut)
        # must be ahead and inside the lane corridor for most of the horizon
        ahead, inside, stations = 0, 0, []
        for k in ks:
            st, lat = station_offset(pts, s, fut[k][:2])
            if st > 0.5:
                ahead += 1
                stations.append(st)
            if abs(lat) <= CORRIDOR_W / 2.0:
                inside += 1
        if ahead < max(2, len(ks) // 2) or inside < max(2, len(ks) // 2):
            continue
        stopped = is_stopped_track(fut)
        g = gap_to_lead(gt, fut, stopped)
        if not np.isfinite(g):
            continue
        # approaching: the gap at the end of the horizon is smaller than at the start
        g_first = _gap_at(gt, fut, ks[0], stopped)
        g_last = _gap_at(gt, fut, ks[-1], stopped)
        closing = np.isfinite(g_first) and np.isfinite(g_last) and (g_last < g_first - 0.2)
        if not closing:
            continue
        key = g - (2.0 if stopped else 0.0)
        if best is None or key < best[0]:
            best = (key, tk, g, stopped, closing)
    return (None, np.nan, False, False) if best is None else (best[1], best[2], best[3], best[4])


def _gap_at(traj, fut, k, stopped):
    eps = ego_polys(traj)
    if k - 1 >= len(eps) or k not in fut:
        return np.nan
    e = eps[k - 1]; b = fut[k]
    a = _rect(b[0], b[1], b[3], b[4], b[6])
    if stopped:
        return e.distance(a) if not e.intersects(a) else -float(np.sqrt(e.intersection(a).area))
    fl = front_line(e)
    return fl.distance(a) if not fl.intersects(a) else -float(np.sqrt(e.intersection(a).area))


def gap_to_lead(traj, fut, stopped):
    """Min front-bumper gap over the horizon; its zero crossing is an at-fault hit."""
    vals = [_gap_at(traj, fut, k, stopped) for k in sorted(fut)]
    vals = [v for v in vals if np.isfinite(v)]
    return min(vals) if vals else np.nan


def classify(traj, fut):
    """Predicted NAVSIM collision type of the first contact (None if no contact)."""
    eps = ego_polys(traj)
    for k in sorted(fut):
        if k - 1 >= len(eps):
            continue
        e = eps[k - 1]; box = fut[k]
        a = _rect(box[0], box[1], box[3], box[4], box[6])
        if not e.intersects(a):
            continue
        prev = traj[k-2, :2] if k >= 2 else np.zeros(2)
        if float(np.hypot(*(traj[k-1, :2] - prev)) / DT) <= STOP_SPEED:
            return "STOPPED_EGO"
        if agent_speed(fut, k) <= STOP_SPEED:
            return "STOPPED_TRACK"
        th = traj[k-1, 2]; d = box[:2] - traj[k-1, :2]
        if float(d[0]*np.cos(th) + d[1]*np.sin(th)) <= 0.0:
            return "ACTIVE_REAR"
        if front_line(e).intersects(a):
            return "ACTIVE_FRONT"
        return "ACTIVE_LATERAL"
    return None


# ------------------------------------------------ longitudinal perturbation ---

def advance_traj(gt, delta, w):
    """Move each waypoint FORWARD ALONG THE EGO'S OWN PATH by delta * w[i].

    The path itself is untouched -- only the station reached at each timestep
    changes -- so the footprint sweep stays inside the same corridor and DAC / LK
    / DDC cannot move. Past the end of the logged path the ego continues straight
    along its final heading, which is what holding speed would do."""
    pts, s = path_frame(gt)
    wp_s = s[::6][1:9]
    new_s = wp_s + delta * w
    out = gt.copy().astype(np.float64)
    L = float(s[-1])
    for i, si in enumerate(new_s):
        if si <= L:
            x = np.interp(si, s, pts[:, 0]); y = np.interp(si, s, pts[:, 1])
            j = int(np.clip(np.searchsorted(s, si), 1, len(s) - 1))
            t = pts[j] - pts[j-1]
            th = np.arctan2(t[1], t[0]) if np.linalg.norm(t) > 1e-6 else gt[i, 2]
        else:                                  # extrapolate along the final heading
            th = float(gt[-1, 2])
            x = pts[-1, 0] + (si - L) * np.cos(th)
            y = pts[-1, 1] + (si - L) * np.sin(th)
        out[i] = (x, y, th)
    return out
