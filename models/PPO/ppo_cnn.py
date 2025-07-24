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

# define models (stochastic and deterministic models) using mixins
class PPO_Policy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum",
                 num_envs=1, arch = [64,64], activation = "relu", vision_h = 32, vision_w = 32, num_robot_obs = 14):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.num_envs = num_envs
        self.policy_arch = arch
        self.activation = activation
        self.device = device
        self.vision_h = vision_h
        self.vision_w = vision_w
        self.num_robot_obs = num_robot_obs
        self.cnn_output = 32

                ### CNN feature extractor set up for medium and target
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



        self.obs_feature_size = self.num_robot_obs + self.cnn_output

        ### MLP Network set

        numl = len(self.policy_arch)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(self.obs_feature_size, self.policy_arch[0], device=self.device))
        for i in range(len(self.policy_arch) - 1):
            self.fc.append(nn.Linear(self.policy_arch[i], self.policy_arch[i+1], device=self.device))
        self.fc.append(nn.Linear(self.policy_arch[numl - 1], self.num_actions, device=self.device))
        
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    #def get_specification(self):
    #    # batch size (N) is the number of envs
    #    return {"rnn": {"sequence_length": self.sequence_length,
    #                    "sizes": [(self.num_layers, self.num_envs, self.hidden_size),    # hidden states (D ∗ num_layers, N, Hout)
    #                              (self.num_layers, self.num_envs, self.hidden_size)]}}  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)
        #tmap = space["tmap"]
        fmap = space["fmap"]
        tool_obs = space["tool"]
        #print("hmap size: " + str(hmap.shape))


        vis = fmap.permute(0, 3, 1, 2) #torch.stack((hmap,tmap))
        vis.to(self.device)
        tool_obs.to(self.device)
        ft = self.feature_extractor(vis)
        x = torch.cat([ft,tool_obs.view(states.shape[0], -1)], dim = -1)

        for i in range(len(self.fc)):
            l = self.fc[i]
            x = l(x)
            if i < len(self.fc) - 1:
                if self.activation == "relu":
                    x = F.relu(x)
    
        return torch.tanh(x), self.log_std_parameter, {}

    

class PPO_Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 num_envs=1, arch = [64,64], activation = "relu", vision_h = 32, vision_w = 32, num_robot_obs = 14):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.num_envs = num_envs
        self.value_arch = arch
        self.activation = activation
        self.device = device
        self.vision_h = vision_h
        self.vision_w = vision_w
        self.num_robot_obs = num_robot_obs
        self.cnn_output = 32

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
        self.obs_feature_size = self.num_robot_obs + self.cnn_output

        numl = len(self.value_arch)
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(self.obs_feature_size, self.value_arch[0], device=self.device))
        for i in range(len(self.value_arch) - 1):
            self.fc.append(nn.Linear(self.value_arch[i], self.value_arch[i+1], device=self.device))
        self.fc.append(nn.Linear(self.value_arch[numl - 1], 1, device=self.device))


    #def get_specification(self):
    #    # batch size (N) is the number of envs
    #    return {"rnn": {"sequence_length": self.sequence_length,
    #                    "sizes": [(self.num_layers, self.num_envs, self.hidden_size)]}}  # hidden states (D ∗ num_layers, N, Hout)

    def compute(self, inputs, role):
        states = inputs["states"]
        space = self.tensor_to_space(states, self.observation_space)
        #tmap = space["tmap"]
        fmap = space["fmap"]
        tool_obs = space["tool"]
        #print("hmap size: " + str(hmap.shape))


        vis = fmap.permute(0, 3, 1, 2) #torch.stack((hmap,tmap))
        vis.to(self.device)
        tool_obs.to(self.device)
        ft = self.feature_extractor(vis)
        x = torch.cat([ft,tool_obs.view(states.shape[0], -1)], dim = -1)

        for i in range(len(self.fc)):
            l = self.fc[i]
            x = l(x)
            if i < len(self.fc) - 1:
                if self.activation == "relu":
                    x = F.relu(x)


        return x, {}
