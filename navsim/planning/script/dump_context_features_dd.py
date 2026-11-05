"""Dump frozen-encoder scene features for the supervised EPDMS scorer --
DiffusionDrive variant.

Runs DiffusionDrive's frozen V2TransfuserModel over a split and saves
{token: keyval} to disk, where `keyval` is the [N_tokens, 65, tf_d_model]
tensor that the trajectory decoder cross-attends to (64 BEV spatial tokens +
1 status token). Same feature as MeanFuser's `dump_context_features_mf.py`.

We capture `keyval` via a forward pre-hook on the model's `_tf_decoder`
(its second positional argument), so the DiffusionDrive model code itself
is not modified.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.planning.training.dataset import CacheOnlyDataset

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _to_device(x, device):
    """Recursively move tensors in nested dict/list structures to a device."""
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: _to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_device(v, device) for v in x]
    return x


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the frozen DD encoder over a split and dump per-token keyval."""
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    # this script never scores (DD's TransfuserAgent has no scorer attached)
    if hasattr(agent, "pdm_score_computer"):
        agent.pdm_score_computer = None

    device = torch.device("cuda")
    # DD's model attribute is _transfuser_model (compare MeanFuser's
    # meanfuser_model). The model is the V2TransfuserModel instance.
    model = agent._transfuser_model.to(device).eval()

    # Forward pre-hook captures the `keyval` argument to _tf_decoder. The call
    # in V2TransfuserModel.forward is `self._tf_decoder(query, keyval)`, so
    # keyval is the second positional argument.
    captured: dict = {}

    def _capture_keyval(module, args):
        if len(args) >= 2:
            captured["keyval"] = args[1]
        else:
            raise RuntimeError(
                f"_tf_decoder called with fewer than 2 positional args: {len(args)}"
            )

    hook = model._tf_decoder.register_forward_pre_hook(_capture_keyval)

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
        logger.info(f"Dumping DD encoder context (keyval) for {len(dataset)} scenes...")

        context_features: dict = {}
        with torch.no_grad():
            for batch_idx, (features, targets) in enumerate(dataloader):
                features = _to_device(features, device)
                _ = model(features, targets=targets)  # triggers the hook
                keyval = captured["keyval"].cpu().numpy()
                for token, kv in zip(targets["token"], keyval):
                    context_features[token] = kv.astype(np.float32)
                if batch_idx % 100 == 0:
                    logger.info(f"  batch {batch_idx} / {len(dataloader)}")
    finally:
        hook.remove()

    save_path = Path(cfg.output_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    out_path = save_path / f"context_features_{timestamp}.pkl"
    pickle.dump(context_features, open(out_path, "wb"))
    logger.info(f"Saved {len(context_features)} context features to {out_path}")


if __name__ == "__main__":
    main()
