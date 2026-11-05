"""Deploy a single STORED proposal (or the ARM) as the agent's output, so the
official one-stage harness scores that exact trajectory under EPDMS.

Used to build the per-proposal EPDMS matrix (ground-truth best-of-N oracle):
run this agent once per proposal index k (and once for the ARM), each run
deploys proposals[k] for every scene; the per-scene max over the runs' EPDMS
is the EPDMS oracle. Proposals are read from the dataset (the same ones the
Q-net / offline sweep use), NOT re-sampled, so every run scores an identical
trajectory set -- immune to stochastic proposal generation.

No model / checkpoint / sensors: it is a pure lookup keyed by the scene's
initial_token. Set agent.deploy_index=<k> or agent.deploy_index=arm.
"""

import pickle
from typing import List

import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory


class DeployIndexAgent(AbstractAgent):
    """Deploys dataset[token]['proposals'][k] (or ['arm_trajectory'])."""

    requires_scene = True

    def __init__(
        self,
        candidate_dataset_path: str,
        deploy_index,  # int k, or the string "arm"
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
    ):
        super().__init__(trajectory_sampling=trajectory_sampling, requires_scene=True)
        self._candidate_dataset_path = candidate_dataset_path
        # normalize deploy_index: "arm" -> arm_trajectory; "gt" -> gt_trajectory
        # (the human driver); otherwise an int proposal index.
        self._mode = "proposal"
        self._k = None
        if isinstance(deploy_index, str) and deploy_index.strip().lower() in ("arm", "gt"):
            self._mode = deploy_index.strip().lower()
        else:
            self._k = int(deploy_index)
        self._rl = None

    def name(self) -> str:
        return self.__class__.__name__

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def initialize(self) -> None:
        with open(self._candidate_dataset_path, "rb") as f:
            self._rl = pickle.load(f)
        tag = {"arm": "ARM", "gt": "GT (human)"}.get(self._mode, f"proposal[{self._k}]")
        print(f"DeployIndexAgent: deploying {tag} from {self._candidate_dataset_path} ({len(self._rl)} tokens)")

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        token = scene.scene_metadata.initial_token
        entry = self._rl[token]
        if self._mode == "arm":
            poses = np.asarray(entry["arm_trajectory"], dtype=np.float32)
        elif self._mode == "gt":
            poses = np.asarray(entry["gt_trajectory"], dtype=np.float32)
        else:
            poses = np.asarray(entry["proposals"][self._k], dtype=np.float32)
        return Trajectory(poses, self._trajectory_sampling)
