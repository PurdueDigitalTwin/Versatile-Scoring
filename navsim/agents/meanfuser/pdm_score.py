import os, lzma, pickle, torch
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hydra
from hydra.utils import instantiate
from hydra.core.global_hydra import GlobalHydra
from navsim.evaluate.pdm_score import pdm_score
from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer


class ComputePDMScore():
    """PDMS (v1) scorer over a metric cache under $NAVSIM_CACHE_ROOT.

    Lazy: the constructor only stores its arguments. The Hydra compose, the
    MetricCacheLoader, simulator and scorer are built on the first
    get_pdm_scores call, so building an agent never touches the metric cache.
    Each token's cache is decompressed on first use and kept; preload=True
    instead loads every cache at that first call.
    """

    def __init__(self, cache_path, preload=False, preload_workers=16):
        self._cache_path = cache_path
        self._preload = preload
        self._preload_workers = preload_workers
        self.metric_cache_loader = None
        self.simulator = None
        self.scorer = None
        self._mc_cache = {}

    def _ensure_initialized(self):
        if self.metric_cache_loader is not None:
            return
        NAVSIM_CACHE_ROOT = os.environ.get("NAVSIM_CACHE_ROOT")
        metric_cache_path = f"{NAVSIM_CACHE_ROOT}/{self._cache_path}"
        if not Path(metric_cache_path).is_dir():
            raise FileNotFoundError(
                f"[ComputePDMScore] metric cache {metric_cache_path} not found "
                f"(NAVSIM_CACHE_ROOT={NAVSIM_CACHE_ROOT}); build it with the v1 metric-caching "
                "step of README.md first")
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        hydra.initialize(config_path="../../planning/script/config/pdm_scoring")
        pdm_score_cfg = hydra.compose(
            config_name="default_run_pdm_score",
            overrides=[f"metric_cache_path={metric_cache_path}"],
        )
        metric_cache_loader = MetricCacheLoader(Path(pdm_score_cfg.metric_cache_path))
        self.simulator: PDMSimulator = instantiate(pdm_score_cfg.simulator)
        self.scorer: PDMScorer = instantiate(pdm_score_cfg.scorer)
        self.metric_cache_loader = metric_cache_loader
        if self._preload:
            self._preload_all(self._preload_workers)

    @staticmethod
    def _load_one_path(path):
        with lzma.open(path, "rb") as f:
            return pickle.load(f)

    def _get_mc(self, token):
        mc = self._mc_cache.get(token)
        if mc is None:
            mc = self._load_one_path(self.metric_cache_loader.metric_cache_paths[token])
            self._mc_cache[token] = mc
        return mc

    def _preload_all(self, workers):
        tokens = list(self.metric_cache_loader.metric_cache_paths.keys())
        paths  = [self.metric_cache_loader.metric_cache_paths[t] for t in tokens]
        print(f"[ComputePDMScore] preloading {len(tokens)} metric caches with {workers} workers...")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for token, mc in zip(tokens, ex.map(self._load_one_path, paths)):
                self._mc_cache[token] = mc
        print(f"[ComputePDMScore] preloaded {len(self._mc_cache)} caches")

    def get_pdm_scores(self, trajectorys, tokens):
        self._ensure_initialized()
        device = trajectorys.device
        trajectorys = trajectorys.cpu().numpy()
        pdm_scores = {}
        for trajectory, token in zip(trajectorys, tokens):
            metric_cache = self._get_mc(token)
            interval_length = 0.1 if trajectory.shape[0] == 40 else 0.5
            traj_obj = Trajectory(
                trajectory,
                TrajectorySampling(time_horizon=4, interval_length=interval_length),
            )
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=traj_obj,
                future_sampling=self.simulator.proposal_sampling,
                simulator=self.simulator,
                scorer=self.scorer,
            ).__dict__
            for k, v in pdm_result.items():
                pdm_scores.setdefault(k, []).append(v)
        return {k: (torch.tensor(v).mean() * 100).to(device) for k, v in pdm_scores.items()}