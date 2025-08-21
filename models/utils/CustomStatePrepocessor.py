from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO as PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


class CustomStatePreprocessor(RunningStandardScaler):
    def forward(
        self, x: torch.Tensor, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor:
        # Don't scale edge_index
        print("states to be scaled: ", x)
        edge_index = x["edge_index"]
        other = {k: v for k, v in x.items() if k != "edge_index"}
        scaled = super().forward(other, train, inverse, no_grad)
        scaled["edge_index"] = edge_index
        return scaled