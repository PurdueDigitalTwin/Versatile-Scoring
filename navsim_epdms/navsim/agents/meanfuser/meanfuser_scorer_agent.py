"""MeanFuser with the trained scorer in the loop, for the official one-stage EPDMS evaluator
(scripts/eval_official.sh). Loads the nine ScorerNet heads comp{j}_<scorer_tag>.pt of one
scripts/train_scorer.sh run, scores the frozen planner's proposals against the encoder's scene
context (`keyval`: 64 BEV tokens + 1 status token), composes EPDMS and drives the argmax. The
planner itself is untouched.
"""
import os
from typing import Dict
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.agents.meanfuser.meanfuser_agent import MeanfuserAgent
from navsim.agents.meanfuser.meanfuser_config import MeanfuserConfig
from navsim.agents.meanfuser.scorer_net import ScorerNet, compose_epdms, EPDMS_N_COMP
from navsim.common.dataclasses import AgentInput, Trajectory


class MeanFuserScorerAgent(MeanfuserAgent):
    def __init__(self, config: MeanfuserConfig, lr: float, max_epochs: int = 150,
                 checkpoint_path: str = None,
                 trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
                 scorer_dir: str = None, scorer_tag: str = "bestval", scorer_d_model: int = 128, **kwargs):
        super().__init__(config, lr, max_epochs, checkpoint_path, trajectory_sampling, **kwargs)
        assert scorer_dir is not None, "agent.scorer_dir must name a scripts/train_scorer.sh run"
        self._scorers = torch.nn.ModuleList()
        for j in range(EPDMS_N_COMP):
            m = ScorerNet(d_model=scorer_d_model)
            m.load_state_dict(torch.load(os.path.join(scorer_dir, f"comp{j}_{scorer_tag}.pt"), map_location="cpu"))
            m.eval(); self._scorers.append(m)

    def initialize(self) -> None:
        sd = torch.load(self._checkpoint_path, map_location="cpu")["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in sd.items()}, strict=False)
        print(f"Loaded MeanFuser from {self._checkpoint_path}")

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        features: Dict[str, torch.Tensor] = {}
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        for k, v in features.items():
            if isinstance(v, torch.Tensor): features[k] = v.unsqueeze(0)
            elif isinstance(v, list):
                for i, t in enumerate(v):
                    if isinstance(t, torch.Tensor): features[k][i] = t.unsqueeze(0)
            elif isinstance(v, str): features[k] = [v]
        with torch.no_grad():
            results = self.forward(features)
            proposals = results["pred_trajectorys"]                      # [1, P, 8, 3]
            keyval = results["encoder_output"]["keyval"]                 # [1, 65, d]
            comps = torch.stack([net(keyval.float(), proposals.float())[0] for net in self._scorers], dim=-1)  # [P, 9]
            best = int(compose_epdms(comps).argmax().item())
            poses = proposals[0, best].cpu().numpy()
        return Trajectory(poses, self._trajectory_sampling)
