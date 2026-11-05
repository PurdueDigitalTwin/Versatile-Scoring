"""Supervised scorer for re-ranking candidate trajectories.

A ScorerNet predicts one EPDMS component of a candidate trajectory given the frozen
planner's scene features (the encoder's keyval: 64 BEV tokens + 1 status token,
from dump_context_features_mf.py / dump_context_features_dd.py). Nine ScorerNets (one
per component) are trained on EPDMS labels and composed via compose_epdms; at
inference the candidates are re-ranked by the composed score.
"""

import torch
import torch.nn as nn

from navsim.agents.meanfuser.utils import HORIZON, ACTION_DIM_ORI


def dsilu(x: torch.Tensor) -> torch.Tensor:
    """dSiLU = d/dx SiLU = sigmoid(x) * (1 + x*(1 - sigmoid(x))). Bounded ~[-0.1, 1.1].
    Best in the activation ablation: dSiLU > GELU > SiLU > ReLU."""
    s = torch.sigmoid(x)
    return s * (1.0 + x * (1.0 - s))


class DSiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return dsilu(x)


class ScorerNet(nn.Module):
    """Predicts one EPDMS component per candidate trajectory from cached scene context."""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
    ):
        # Scoring is always INDEPENDENT: candidates never attend to each other
        # (diagonal self-attn mask), so each is scored from its own trajectory plus
        # the scene alone. This makes scores set-size-invariant -- adding or removing
        # candidates cannot change the others' scores.
        super().__init__()
        self.traj_encoder = nn.Sequential(
            nn.Linear(HORIZON * ACTION_DIM_ORI, d_model),
            DSiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation=dsilu,
        )
        self.cross_attn = nn.TransformerDecoder(decoder_layer, num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            DSiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, context: torch.Tensor, actions: torch.Tensor,
                return_logits: bool = False) -> torch.Tensor:
        """
        :param context: [B, num_context, d_model] cached encoder keyval features
        :param actions: [B, N, HORIZON, 3] candidate trajectories (x, y, heading)
        :param return_logits: return the pre-sigmoid score
        :return: [B, N] predicted component in (0, 1), or the raw logit
        """
        batch_size, num_actions = actions.shape[:2]
        traj_emb = self.traj_encoder(actions.reshape(batch_size, num_actions, -1))
        # block cross-candidate self-attention: each candidate attends only to itself
        tgt_mask = torch.full((num_actions, num_actions), float("-inf"), device=traj_emb.device)
        tgt_mask.fill_diagonal_(0.0)
        attn_out = self.cross_attn(traj_emb, context, tgt_mask=tgt_mask)
        q = self.head(attn_out).squeeze(-1)
        return q if return_logits else torch.sigmoid(q)


# ---- EPDMS (the deployed NAVSIMv2 metric) composition. 9 sub-components, order
# matches tools/dump_components.COMP:
#   [NC, DAC, DDC, TLC, EP, TTC, LK, HC, TFC]
# EPDMS = (NC*DAC*DDC*TLC) * (5*EP + 5*TTC + 2*LK + 2*HC + 2*TFC) / 16 ----

EPDMS_NAMES = ["NC", "DAC", "DDC", "TLC", "EP", "TTC", "LK", "HC", "TFC"]
EPDMS_N_COMP = 9
EPDMS_N_GATES = 4  # NC, DAC, DDC, TLC are the multiplicative gates
EPDMS_WEIGHTS = (5.0, 5.0, 2.0, 2.0, 2.0)  # EP, TTC, LK, HC, TFC


def compose_epdms(comp: torch.Tensor) -> torch.Tensor:
    """EPDMS from its 9 sub-components. Composes the FULL weight denominator (16);
    intended for PREDICTED components, which are always valid (no NaN). The
    scorer renormalizes when a true weighted metric is unavailable -- that NaN
    handling lives in the offline label tooling, not here.
    :param comp: [..., 9] = [NC,DAC,DDC,TLC,EP,TTC,LK,HC,TFC] in (0,1) -> [...]."""
    gate = comp[..., 0] * comp[..., 1] * comp[..., 2] * comp[..., 3]
    w = comp.new_tensor(EPDMS_WEIGHTS)
    weighted = (comp[..., 4:9] * w).sum(-1) / w.sum()
    return gate * weighted
