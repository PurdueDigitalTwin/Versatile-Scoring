# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
from typing import Any, Union, Dict, List
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LRScheduler, OneCycleLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.meanfuser.meanfuser_features import MFFeatureBuilder, MFTargetBuilder
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.agents.meanfuser.meanfuser_model import MeanfuserModel
from navsim.agents.meanfuser.meanfuser_config import MeanfuserConfig


class PeriodicFullEval(pl.Callback):
    """Run a full validation every `full_every` epochs (and on the final epoch);
    skip validation entirely on all other epochs.
    """
    def __init__(self, full_every: int = 3):
        self.full_every = full_every
        self._saved_limit = None

    def on_train_start(self, trainer, pl_module):
        # Snapshot whatever the user configured so we can restore it on full-eval epochs.
        self._saved_limit = trainer.limit_val_batches

    def on_train_epoch_start(self, trainer, pl_module):
        epoch_one_indexed = trainer.current_epoch + 1
        is_full_period = (epoch_one_indexed % self.full_every) == 0
        is_last_epoch = epoch_one_indexed >= trainer.max_epochs
        if is_full_period or is_last_epoch:
            trainer.limit_val_batches = self._saved_limit if self._saved_limit else 1.0
            print(f"[PeriodicFullEval] FULL eval at epoch {trainer.current_epoch}", flush=True)
        else:
            trainer.limit_val_batches = 0.0
            print(f"[PeriodicFullEval] validation skipped at epoch {trainer.current_epoch}", flush=True)
        # PL 2.2.x caches the resolved val-batch count in the val loop after the
        # first setup_data(); nulling _combined_loader forces it to re-run so the
        # new limit_val_batches takes effect this epoch.
        trainer.fit_loop.epoch_loop.val_loop._combined_loader = None


class MeanfuserAgent(AbstractAgent):
    def __init__(
            self,
            config: MeanfuserConfig,
            lr: float,
            max_epochs: int = 150,
            checkpoint_path: str = None,
            full_eval_every: int = 3,
            **kwargs,
    ):
        super().__init__()
        self._config = config
        self._lr = lr
        self._checkpoint_path = checkpoint_path
        self.meanfuser_model = MeanfuserModel(config)
        self.max_epochs = max_epochs
        self._full_eval_every = full_eval_every

        from navsim.agents.meanfuser.pdm_score import ComputePDMScore
        # Validation PDMS scorer. Lazy: nothing is read from the navtest metric cache
        # until the first validation batch is scored. MF_PDM_PRELOAD=1 opts in to
        # loading all ~12k caches at that first call (tens of GB of RAM, training only).
        # Setting agent.pdm_score_computer = None disables scoring.
        _preload = os.environ.get("MF_PDM_PRELOAD", "0") == "1"
        self.pdm_score_computer = ComputePDMScore(cache_path='navtest_v1_metric_cache', preload=_preload)

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=True)
        print(f"Loaded model from {self._checkpoint_path}")

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig(
            cam_f0=[3],
            cam_l0=[3],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[3],
            cam_r1=[3],
            cam_r2=[],
            cam_b0=[],
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [MFTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [MFFeatureBuilder(config=self._config)]

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        return self.meanfuser_model(features, targets)

    def compute_loss(self, features, targets, predictions, tokens=None):
        loss_dict = self.meanfuser_model.get_meanfuser_loss(predictions, targets)

        if not self.training:
            if self.pdm_score_computer is not None:
                pdm_scores = self.pdm_score_computer.get_pdm_scores(
                    predictions["trajectory"], targets["token"]
                )
                loss_dict.update(pdm_scores)
            # Detach every tensor so nothing returned has a live autograd graph.
            for k, v in list(loss_dict.items()):
                if torch.is_tensor(v) and v.requires_grad:
                    loss_dict[k] = v.detach()

        return loss_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        
        if self._config.scheduler_type == 'constant':
            return torch.optim.Adam(self.meanfuser_model.parameters(), lr=self._lr)
        elif self._config.scheduler_type == 'warmup_cos':
            backbone_params_name = '_backbone.image_encoder'
            img_backbone_params = list(
                filter(lambda kv: backbone_params_name in kv[0], self.meanfuser_model.named_parameters()))
            default_params = list(
                filter(lambda kv: backbone_params_name not in kv[0], self.meanfuser_model.named_parameters()))
            params_lr_dict = [
                {'params': [tmp[1] for tmp in default_params]},
                {
                    'params': [tmp[1] for tmp in img_backbone_params],
                    'lr': self._lr * self._config.lr_mult_backbone,
                    'weight_decay': self._config.backbone_wd
                }
            ]
            
            optim = torch.optim.AdamW(params_lr_dict, lr=self._lr, weight_decay=self._config.weight_decay)
            scheduler = WarmupCosLR(
                optim,
                min_lr=1e-7,
                epochs=self.max_epochs,
                warmup_epochs=3,
            )
            return{
                "optimizer": optim,
                "lr_scheduler": scheduler
            }
        else:
            raise ValueError('Unsupported lr scheduler')

    def get_training_callbacks(self) -> List[pl.Callback]:
        val_score = "val_score"
        filename = "{epoch:02d}-{val_score:.4f}"
        mode = 'max'

        ckpt_callback = ModelCheckpoint(
            save_top_k=10,
            monitor=val_score,
            mode=mode,
            dirpath=f"{self._config.output_dir}/",
            filename=filename,
            # save only after a validation run; PeriodicFullEval skips most epochs
            save_on_train_epoch_end=False,
        )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
        periodic_eval = PeriodicFullEval(full_every=self._full_eval_every)
        return [ckpt_callback, lr_monitor, periodic_eval]


class WarmupCosLR(_LRScheduler):
    def __init__(
        self, optimizer, min_lr, warmup_epochs, epochs, last_epoch=-1, verbose=False
    ) -> None:
        self.min_lr = min_lr
        self.lr = optimizer.param_groups[0]["lr"]
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        super(WarmupCosLR, self).__init__(optimizer, last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_init_lr(self):
        lr = self.lr / self.warmup_epochs
        return lr

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            lr = self.lr * (self.last_epoch + 1) / self.warmup_epochs
        else:
            lr = self.min_lr + 0.5 * (self.lr - self.min_lr) * (
                1
                + math.cos(
                    math.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.epochs - self.warmup_epochs)
                )
            )
        if "lr_scale" in self.optimizer.param_groups[0]:
            return [lr * group["lr_scale"] for group in self.optimizer.param_groups]

        return [lr for _ in self.optimizer.param_groups]
