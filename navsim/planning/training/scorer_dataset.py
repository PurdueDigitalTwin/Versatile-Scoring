"""Candidate dataset for training the supervised EPDMS component scorer.

Pairs the frozen planner's cached scene features (dump_context_features_mf.py for MeanFuser,
dump_context_features_dd.py for DiffusionDrive) with the per-scene candidate trajectories
(build_plain_dataset.py, or a designed set from tools/make_trainsets.py) and their 9-component
EPDMS labels (tools/dump_components.py or make_trainsets.py). The "rewards" key of a sample is
the composed EPDMS label of each candidate.
"""

import pickle
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class ScorerDataset(Dataset):
    """Yields cached scene context, candidate trajectories and EPDMS labels."""

    def __init__(self, candidates_path, ctx_path, labels_path, include_human: bool = False,
                 tokens=None):
        """
        :param candidates_path: candidate pickle (proposals + gt_trajectory per token) from
            scripts/dump_candidates.sh or tools/make_trainsets.py, or an already-loaded dict
        :param ctx_path: scene-context pickle ({token: [65, d_model]}) from
            dump_context_features_mf.py or dump_context_features_dd.py, or an already-loaded dict
        :param labels_path: 9-component EPDMS labels from tools/dump_components.py
            ({"data": {token: {components [N_keys,9], epdms [N_keys]}},
              "keys": [0..N-1, "human", ...]}), or an already-loaded dict. Proposals come
            from the dataset, so the actions are byte-identical to the deploy sweep.
            Integer keys are the selectable proposals; named keys are reference rows and
            are NEVER part of `num_proposals`. ("arm" may still be present as the
            eval-only baseline column; the scorer never trains on it.)
        :param include_human: append the HUMAN (logged GT) trajectory as an extra
            action/label. It is a training target only -- excluded from the re-rank
            argmax pool, exactly as the ARM row used to be.
        :param tokens: optional explicit token subset (train/val splits)
        """
        load = lambda p: p if isinstance(p, dict) else pickle.load(open(p, "rb"))
        self._candidates = load(candidates_path)
        self._ctx = load(ctx_path)
        blob = load(labels_path)
        self._labels = blob["data"]
        keys = list(blob["keys"])
        # named keys may appear in any order/count ("human", "arm"), so count the
        # integer proposal keys rather than assuming the extras are last.
        self._n_props = sum(isinstance(k, (int, np.integer)) for k in keys)
        self._human_idx = keys.index("human") if "human" in keys else None
        self._keys = keys
        if include_human and self._human_idx is None:
            raise KeyError(
                f"--human requested but the labels pkl has no 'human' key (keys={keys}). "
                "scripts/label_slots.sh mf <split> labels the gt slot and stitches it in.")
        self.include_human = include_human

        self.tokens = sorted(set(self._candidates) & set(self._ctx) & set(self._labels))
        if tokens is not None:
            self.tokens = sorted(set(self.tokens) & set(tokens))

        # proposals only -- the extra human row is appended after this count
        self.num_proposals = int(self._candidates[self.tokens[0]]["proposals"].shape[0])

    def subset(self, tokens) -> "ScorerDataset":
        """Same data, restricted to `tokens` (shares the loaded pickles)."""
        return ScorerDataset(self._candidates, self._ctx,
                            {"data": self._labels, "keys": self._keys},
                            include_human=self.include_human, tokens=tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        token = self.tokens[idx]
        actions = self._candidates[token]["proposals"]
        lab = self._labels[token]
        rewards = np.asarray(lab["epdms"][: self._n_props], dtype=np.float32)
        comps = np.asarray(lab["components"][: self._n_props], dtype=np.float32)
        if self.include_human:
            actions = np.concatenate([actions, self._candidates[token]["gt_trajectory"][None]], axis=0)
            rewards = np.concatenate([rewards, np.asarray(lab["epdms"][self._human_idx])[None]], axis=0)
            comps = np.concatenate([comps, np.asarray(lab["components"][self._human_idx])[None]], axis=0)
        return {
            "context": torch.from_numpy(np.asarray(self._ctx[token], dtype=np.float32)),
            "actions": torch.from_numpy(np.asarray(actions, dtype=np.float32)),
            "rewards": torch.from_numpy(rewards),
            "components": torch.from_numpy(comps),
        }
