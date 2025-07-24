import torch
import torch.nn as nn

# Import the skrl components to build the RL system
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.torch import wrap_env
import torch.nn.functional as F


# Define the models (stochastic and deterministic models) for the SAC agent using the mixins.
# - StochasticActor (policy): takes as input the environment's observation/state and returns an action
# - Critic: takes the state and action as input and provides a value to guide the policy
class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, arch = [64,64], activation = "relu"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.policy_arch = arch
        self.activation = activation
        self.device = device

        numl = len(self.policy_arch)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(self.num_observations, self.policy_arch[0], device=self.device))
        for i in range(len(self.policy_arch) - 1):
            self.fc.append(nn.Linear(self.policy_arch[i], self.policy_arch[i+1], device=self.device))
        self.fc.append(nn.Linear(self.policy_arch[numl - 1], self.num_actions, device=self.device))

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        states = inputs["states"]

        space = self.tensor_to_space(states, self.observation_space)

        x = space["tool"].view(states.shape[0], -1)
        x.to(self.device)
        for i in range(len(self.fc)):
            l = self.fc[i]
            x = l(x)
            if i < len(self.fc) - 1:
                if self.activation == "relu":
                    x = F.relu(x)
        return torch.tanh(x), self.log_std_parameter, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, arch = [64,64], activation = "relu"):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)


        self.num_envs = num_envs
        self.policy_arch = arch
        self.activation = activation
        self.device = device

        numl = len(self.policy_arch)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(self.num_observations, self.policy_arch[0], device=self.device))
        for i in range(len(self.policy_arch) - 1):
            self.fc.append(nn.Linear(self.policy_arch[i], self.policy_arch[i+1], device=self.device))
        self.fc.append(nn.Linear(self.policy_arch[numl - 1], 1, device=self.device))

    def compute(self, inputs, role):
        states = inputs["states"]

        space = self.tensor_to_space(states, self.observation_space)

        x = space["tool"].view(states.shape[0], -1)
        x.to(self.device)
        for i in range(len(self.fc)):
            l = self.fc[i]
            x = l(x)
            if i < len(self.fc) - 1:
                if self.activation == "relu":
                    x = F.relu(x)

        return x, {}

