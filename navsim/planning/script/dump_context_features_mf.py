"""Dump the frozen MeanFuser encoder's scene context per scene: the `keyval` memory the trajectory
decoder attends to (64 BEV tokens + 1 status token), as {token: [65, d_model]}.

This is the scene feature the scorer reads (run_scorer_training.py --ctx); the DiffusionDrive
scorer reads the same kind of feature from dump_context_features_dd.py. No map or label
information, only the planner's own perception.

Captured via a forward pre-hook on encoder._tf_decoder (called as _tf_decoder(query, keyval) in
meanfuser_model.py), so keyval = the 2nd positional arg (after pos-emb).
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import CacheOnlyDataset
from hydra.utils import instantiate

logger = logging.getLogger(__name__)
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    return x


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the frozen encoder over a split and dump per-token keyval (BEV+status)."""
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    agent.pdm_score_computer = None  # this script never scores

    device = torch.device("cuda")
    model = agent.meanfuser_model.to(device).eval()

    # forward pre-hook: capture keyval = 2nd positional arg of encoder._tf_decoder
    captured: dict = {}

    def _capture(module, args):
        if len(args) >= 2:
            captured["keyval"] = args[1]
        else:
            raise RuntimeError(f"_tf_decoder called with {len(args)} positional args (<2)")

    # model.encoder is a bound method (self=model); the _tf_decoder module lives on model
    hook = model._tf_decoder.register_forward_pre_hook(_capture)

    try:
        scene_filter = instantiate(cfg.train_test_split.scene_filter)
        dataset = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=scene_filter.log_names,
            is_training=False,
        )
        dataloader = DataLoader(dataset, **cfg.dataloader.params, shuffle=False)
        logger.info(f"Dumping encoder keyval (BEV+status) for {len(dataset)} scenes...")

        feats = {}
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(dataloader):
                features = _to_device(features, device)
                _ = model.encoder(features, targets)  # triggers the hook
                keyval = captured["keyval"].cpu().numpy()
                for token, kv in zip(targets["token"], keyval):
                    feats[token] = kv.astype("float32")
                if batch_idx % 100 == 0:
                    logger.info(f"  batch {batch_idx} / {len(dataloader)}")
    finally:
        hook.remove()

    save_path = Path(cfg.output_dir); save_path.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out = save_path / f"context_features_{ts}.pkl"
    pickle.dump(feats, open(out, "wb"))
    logger.info(f"Saved the scene context of {len(feats)} scenes to {out}")


if __name__ == "__main__":
    main()
