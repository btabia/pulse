import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO as PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
import torch.nn.functional as F
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from models.utils.cvc.cvc import CoordConv

# define the shared model
class Shared_PPO_CNN(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", 
                num_envs=1, arch = [64,64], activation = "relu", vision_h = 32, vision_w = 32, num_robot_obs = 14):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions = False, clip_log_std = True, min_log_std = -20, max_log_std=2, reduction="sum", role="policy")
        DeterministicMixin.__init__(self, clip_actions = False, role="value")

        #self.activation = activation
        self.device = device
        self.num_robot_obs = num_robot_obs
        self.cnn_output = 32

        # shared layers/network
        self.feature_extractor = nn.Sequential(
            CoordConv(1, 32, kernel_size = 3, padding = 1, stride=2 ), 
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size =  2),
            CoordConv(32, 64, kernel_size = 3, padding = 1, stride=2), 
            nn.LeakyReLU(0.2),
            CoordConv(64, 64, kernel_size = 3, padding = 1, stride=2), 
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size =  2),
            CoordConv(64, 128, kernel_size = 3, padding = 1, stride=2), 
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128,self.cnn_output ),
            nn.Tanh()
        )

        ### MLP Network set

        self.obs_feature_size = self.num_robot_obs + 32

        self.policy_layer = nn.Sequential(
            nn.Linear(self.obs_feature_size, 512, device=self.device), 
            nn.ReLU(),
            nn.Linear(512, 256, device=self.device), 
            nn.ReLU(),
            nn.Linear(256, 128, device=self.device), 
            nn.ReLU(),
            nn.Linear(128, self.num_actions, device=self.device), 
        )

        self.value_layer = nn.Sequential(
            nn.Linear(self.obs_feature_size, 512, device=self.device), 
            nn.ReLU(),
            nn.Linear(512, 256, device=self.device), 
            nn.ReLU(),
            nn.Linear(256, 128, device=self.device), 
            nn.ReLU(),
            nn.Linear(128, 1, device=self.device), 
        )
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    # override the .act(...) method to disambiguate its call
    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    # forward the input to compute model output according to the specified role
        
    def compute(self, inputs, role):
        if role == "policy":
            states = inputs["states"]
            space = self.tensor_to_space(states, self.observation_space)
            fmap = space["fmap"]
            vis = fmap.permute(0, 3, 1, 2)
            tool_obs = space["tool"]
            vis.to(self.device)
            tool_obs.to(self.device)
            ft = self.feature_extractor(vis)
            self._shared_output = torch.cat([ft,tool_obs.view(states.shape[0], -1)], dim = -1)
            output = torch.tanh(self.policy_layer(self._shared_output))
            return output, self.log_std_parameter, {}

        elif role == "value":
            if self._shared_output == None: 
                states = inputs["states"]
                space = self.tensor_to_space(states, self.observation_space)
                fmap = space["fmap"]
                vis = fmap.permute(0, 3, 1, 2)
                tool_obs = space["tool"]
                vis.to(self.device)
                tool_obs.to(self.device)
                ft = self.feature_extractor(vis)
                shared_output = torch.cat([ft,tool_obs.view(states.shape[0], -1)], dim = -1)
            else: 
                shared_output =self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}


