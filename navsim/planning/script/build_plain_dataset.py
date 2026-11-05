"""Build the candidate dataset for the supervised scorer.

Loads a predictions pickle (per token: ARM trajectory + proposals + GT, as
dumped by run_pdm_score_gpu_v1.py) and PDMS-scores (v1) every trajectory. Writes
a token-keyed dataset of (proposals, proposal_scores, arm/gt trajectory + PDMS).
CPU only. The `*_reward` keys are legacy names for these PDMS values.

The scorer is trained on 9-component EPDMS labels from tools/dump_components.py,
not on the PDMS stored here, and reads cached keyval scene features
(dump_context_features_mf.py / dump_context_features_dd.py); neither is stored here.
"""

import logging
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.utils.multithreading.worker_utils import worker_map
from omegaconf import DictConfig

from navsim.common.dataclasses import Trajectory
from navsim.common.dataloader import MetricCacheLoader
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

CHUNK_SIZE = 64  # tokens per worker task


# PDMS sub-metrics that the score is built from (driving_direction has weight 0
# in PDMS so it is omitted): PDMS = NC*DAC*(5*EP + 5*TTC + 2*C)/12.
COMP_FIELDS = ["no_at_fault_collisions", "drivable_area_compliance",
               "ego_progress", "time_to_collision_within_bound", "comfort"]


def _score(metric_cache, trajectory, simulator: PDMSimulator, scorer: PDMScorer):
    """PDMS aggregate score + the per-component sub-metrics for one trajectory.

    Returns (score, components[5]) where components = [NC, DAC, EP, TTC, comfort].
    The scorer already computes these; we just keep them (free)."""
    r = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
    )
    comps = np.array([getattr(r, f) for f in COMP_FIELDS], dtype=np.float32)
    return float(r.score), comps


def run_pdm_score(args: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Worker: PDMS-score the ARM trajectory, every proposal and GT for a chunk."""
    cfg: DictConfig = args[0]["cfg"]

    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    results: List[Dict[str, Any]] = []
    for a in args:
        predictions = a["model_trajectory"]
        for token in a["tokens"]:
            row: Dict[str, Any] = {"token": token, "valid": True}
            try:
                if token not in metric_cache_loader.tokens:
                    row["valid"] = False
                    results.append(row)
                    continue
                metric_cache = metric_cache_loader.get_from_token(token)

                row["arm_reward"], row["arm_components"] = _score(
                    metric_cache, predictions[token]["trajectory"], simulator, scorer
                )
                row["gt_reward"], row["gt_components"] = _score(
                    metric_cache, predictions[token]["gt_trajectory"], simulator, scorer
                )

                proposals = predictions[token]["pred_trajectorys_array"]
                proposal_scores: List[float] = []
                proposal_components: List[Any] = []
                for i in range(proposals.shape[0]):
                    poses = proposals[i]
                    interval_length = 0.1 if poses.shape[0] == 40 else 0.5
                    prop_traj = Trajectory(
                        poses, TrajectorySampling(time_horizon=4, interval_length=interval_length)
                    )
                    s, c = _score(metric_cache, prop_traj, simulator, scorer)
                    proposal_scores.append(s)
                    proposal_components.append(c)
                row["proposal_scores"] = proposal_scores
                row["proposal_components"] = np.stack(proposal_components)  # [N, 5]
            except Exception:
                logger.warning(f"----------- scoring failed for token {token}:")
                traceback.print_exc()
                row["valid"] = False
            results.append(row)
    return results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Score a predictions pickle into a token-keyed candidate dataset."""
    build_logger(cfg)

    pkl_path = cfg.predictions_pkl
    logger.info(f"Loading predictions from {pkl_path}")
    merged_predictions: Dict[str, Any] = pickle.load(open(pkl_path, "rb"))
    tokens = list(merged_predictions.keys())
    logger.info(f"Loaded {len(tokens)} token predictions; PDMS-scoring proposals...")

    data_points = [
        {
            "cfg": cfg,
            "tokens": tokens[i : i + CHUNK_SIZE],
            "model_trajectory": {t: merged_predictions[t] for t in tokens[i : i + CHUNK_SIZE]},
        }
        for i in range(0, len(tokens), CHUNK_SIZE)
    ]
    worker = build_worker(cfg)
    score_rows: List[Dict[str, Any]] = worker_map(worker, run_pdm_score, data_points)

    dataset: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []
    for row in score_rows:
        if not row["valid"]:
            continue
        token = row["token"]
        pred = merged_predictions[token]
        proposal_scores = np.asarray(row["proposal_scores"], dtype=np.float32)
        dataset[token] = {
            "proposals": np.asarray(pred["pred_trajectorys_array"], dtype=np.float32),
            "proposal_scores": proposal_scores,
            "proposal_components": np.asarray(row["proposal_components"], dtype=np.float32),  # [N,5] NC,DAC,EP,TTC,C
            "arm_trajectory": np.asarray(pred["trajectory"].poses, dtype=np.float32),
            "arm_reward": np.float32(row["arm_reward"]),
            "arm_components": np.asarray(row["arm_components"], dtype=np.float32),
            "gt_trajectory": np.asarray(pred["gt_trajectory"].poses, dtype=np.float32),
            "gt_reward": np.float32(row["gt_reward"]),
            "gt_components": np.asarray(row["gt_components"], dtype=np.float32),
        }
        summary_rows.append(
            {
                "token": token,
                "arm_reward": row["arm_reward"],
                "gt_reward": row["gt_reward"],
                "proposal_best": float(proposal_scores.max()),
                "proposal_mean": float(proposal_scores.mean()),
                "proposal_worst": float(proposal_scores.min()),
            }
        )

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    dataset_path = save_path / f"dataset_{timestamp}.pkl"
    pickle.dump(dataset, open(dataset_path, "wb"))
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(save_path / f"dataset_summary_{timestamp}.csv", index=False)

    n_failed = len(score_rows) - len(dataset)
    logger.info(
        f"""
        ===== candidate dataset built =====
        Tokens written : {len(dataset)} (failed/skipped: {n_failed})
        Mean GT PDMS             : {summary_df["gt_reward"].mean():.4f}
        Mean ARM PDMS            : {summary_df["arm_reward"].mean():.4f}
        Mean proposal PDMS       : {summary_df["proposal_mean"].mean():.4f}
        Mean best-proposal PDMS  : {summary_df["proposal_best"].mean():.4f}
        Mean worst-proposal PDMS : {summary_df["proposal_worst"].mean():.4f}
        Dataset : {dataset_path}
        Summary : {save_path / f"dataset_summary_{timestamp}.csv"}
        ===================================
        """
    )


if __name__ == "__main__":
    main()
