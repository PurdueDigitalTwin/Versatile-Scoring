"""Paths and geometry helpers shared by the sample generators.

  map_api_for     cached nuPlan map API per map location
  drivable_union  union of the drivable-area polygons within `radius` of (cx, cy)
  densify         [8,3] waypoints -> dense polyline with the origin prepended + arc length
  REPO / WS       this repo / the workspace that holds dataset/ and cache/
  EPDMS_ROOT      navsim_epdms/, the official one-stage EPDMS package
  CACHE_ROOT      $NAVSIM_WORKSPACE/cache
  CK              exp/meanfuser_checkpoints
  EGO_L / EGO_W   nuPlan Pacifica footprint, metres

Paths follow scripts/env.sh: NAVSIM_WORKSPACE defaults to the directory that
contains this repo, and the data, map and exp roots derive from it. Importing this
module exports them, so import it before navsim or nuplan.

Imported by the sample generators (tools/make_lateral_samples.py,
tools/longitudinal_geometry.py, tools/make_longitudinal_samples.py,
tools/make_random_samples.py).
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WS = Path(os.environ.get("NAVSIM_WORKSPACE") or REPO.parent).resolve()
EPDMS_ROOT = REPO / "navsim_epdms"
CACHE_ROOT = WS / "cache"
os.environ["NAVSIM_WORKSPACE"] = str(WS)
os.environ["OPENSCENE_DATA_ROOT"] = str(WS / "dataset")
os.environ["NUPLAN_MAPS_ROOT"] = str(WS / "dataset" / "maps")
os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NAVSIM_CACHE_ROOT"] = str(CACHE_ROOT)
os.environ["NAVSIM_EXP_ROOT"] = str(REPO / "exp")

import numpy as np
from shapely.ops import unary_union

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

CK = REPO / "exp" / "meanfuser_checkpoints"
EGO_L, EGO_W = 5.176, 2.297
LAYERS = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR,
          SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA]

_map_cache = {}


def map_api_for(location):
    if location not in _map_cache:
        _map_cache[location] = get_maps_api(os.environ["NUPLAN_MAPS_ROOT"],
                                            os.environ["NUPLAN_MAP_VERSION"], location)
    return _map_cache[location]


def drivable_union(map_api, cx, cy, radius=45.0):
    objs = map_api.get_proximal_map_objects(Point2D(cx, cy), radius, LAYERS)
    polys = [o.polygon for layer in LAYERS for o in objs.get(layer, [])]
    return unary_union(polys) if polys else None


def densify(traj, k=6):
    """[8,3] -> dense polyline with origin prepended; returns pts [M,2], cum arc s."""
    p = np.concatenate([np.zeros((1, 2)), traj[:, :2]], 0)
    t = np.arange(len(p)); ti = np.linspace(0, len(p) - 1, k * (len(p) - 1) + 1)
    pts = np.stack([np.interp(ti, t, p[:, 0]), np.interp(ti, t, p[:, 1])], -1)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return pts, np.concatenate([[0], np.cumsum(ds)])
