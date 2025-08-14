import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
import torch_geometric.transforms as T
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
from skrl.utils.spaces.torch import unflatten_tensorized_space
from models.utils.cvc.cvc import CoordConv

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.transforms import SamplePoints, KNNGraph, GridSampling
from torch_geometric.nn import global_max_pool, global_mean_pool, global_add_pool

class MyData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == 'foo':
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


# define the shared model
class Shared_PPO_GAT(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, device_output,  clip_actions=False,
                clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum", 
                num_envs=1, arch = [64,64], activation = "relu", vision_h = 32, vision_w = 32, num_robot_obs = 14):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions = False, clip_log_std = True, min_log_std = -20, max_log_std=2, reduction="sum", role="policy")
        DeterministicMixin.__init__(self, clip_actions = False, role="value")

        #self.activation = activation
        self.num_envs = 1
        self.device = device
        self.device_output = device_output
        self.num_robot_obs = num_robot_obs
        self.cnn_output = 32
        self.dropout_prob = 0
        self.gat_hidden_features = 128
        self.gat_output_features = 32
        self.lstm_output_features = 32
        self.lstm_hidden_size = 64
        self.sequence_length = 128
        self.lstm_num_layers = 1
        self.obs_feature_size = 2837 # self.num_robot_obs + self.lstm_output_features

        ##### GAT config #####
        self.gat_in_channels = 8
        self.gat_hidden_channels_1 = 32
        self.gat_hidden_channels_2 = 64
        self.gat_out_channels = 64

        self.gat_heads_1 = 4
        self.gat_heads_2 = 1
        self.edge_attr_dim = 3  # Assuming edge attributes are 3-dimensional



        # shared layers/network

        self.pre_process = nn.Sequential(
            nn.Linear(3, 8, device=self.device), 
            nn.LeakyReLU(0.2),
        )

        # layer to extract grid features
        self.gat_conv_1 =  GATv2Conv(
                in_channels = self.gat_in_channels, # variable graph size
                out_channels = self.gat_hidden_channels_1, #
                heads = self.gat_heads_1, 
                edge_dim = self.edge_attr_dim,  # <-- Add this!
                )
        # layer to extract agent features
        self.gat_conv_2 =  GATv2Conv(
                in_channels = self.gat_hidden_channels_1 * self.gat_heads_1 , # variable graph size
                out_channels = self.gat_hidden_channels_2, #
                heads = self.gat_heads_2, 
                edge_dim = self.edge_attr_dim,  # <-- Add this!
                )


        self.post_process = nn.Sequential(
            nn.Linear(128, 64, device=self.device), 
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32, device=self.device), 
            nn.Tanh(),
        )
        
        ### MLP Network set

        self.obs_feature_size = 3 + 32


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
            print("states shape: " + str(states.shape))
            print("states: " + str(states))
            space = unflatten_tensorized_space(self.observation_space, states)
            point_cloud = space["point_cloud"]
            edge_index = space["edge_index"]
            edge_attr = space["edge_attribute"]

            point_cloud = point_cloud.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_attr.to(self.device)
            pc_ = point_cloud

            edge_index = edge_index.type(torch.int64)

            tool_obs = space["tool"]
            tool_obs = tool_obs.to(self.device)
        
            # set the edges to zero between the point cloud and the tool and one within the point cloud
            #edge_attributes_1 = torch.zeros((batch.edge_index.shape[1], 1), device=self.device)
            #edge_attributes_2 = torch.zeros((batch.edge_index.shape[1], 1), device=self.device)
            #edge_attributes_1[0:((4096*4)-1)] = edge_attr[0:((4096*4)-1)]
            #edge_attributes_2[4096:] = edge_attr[4096:]

            print("edge index model : " + str(edge_index))
            print("tool obs model : " + str(tool_obs))
            print("point cloud model : " + str(pc_))
            print("observation space: " + str(self.observation_space))

            # encode the point cloud
            x = self.pre_process(pc_)

            data_list_1 = [Data(x=x[i], edge_index=edge_index[i], edge_attr = edge_attr[i]) for i  in range(states.shape[0])] 
            loader_1 = DataLoader(data_list_1, batch_size = states.shape[0])
            batch_1 = next(iter(loader_1))

            # set the edges attributes to zero between the point cloud and the tool and one within the point cloud
            #x1 = x.view(-1, x.shape[-1])

            x = self.gat_conv_1(x = batch_1.x , edge_index = batch_1.edge_index, edge_attr = batch_1.edge_attr)
            x = F.leaky_relu(x, negative_slope=0.2)
            # set the edges to one between the point cloud and the tool and zero within the point cloud
            x = self.gat_conv_2(x = x, edge_index = batch_1.edge_index, edge_attr = batch_1.edge_attr)
            x = F.leaky_relu(x, negative_slope=0.2)


            #extract the target and tool features from the point cloud
            # capture the target and tool features from the point cloud batch
            x = x.view(states.shape[0], 4098, -1)  # 4096 for the point cloud, 1 for the target, and 1 for the tool
            target_features = x[:, 4096, :]
            tool_features = x[:, 4097, :]
            # concatenate the target and tool features with the tool observations
            x = torch.cat([target_features, tool_features], dim=-1)

            x = self.post_process(x)

            self._gat_output = torch.cat([x.view(states.shape[0], -1),tool_obs.view(states.shape[0], -1)], dim = -1)

            output = torch.tanh(self.policy_layer(self._gat_output))
            output = output.to(self.device_output)
            return output, self.log_std_parameter, {}

        elif role == "value":
            if self._gat_output == None: 
                states = inputs["states"]
                space = unflatten_tensorized_space(self.observation_space, states)
                point_cloud = space["point_cloud"]
                edge_index = space["edge_index"]
                #edge_attr = space["edge_attribute"]

                point_cloud.to(self.device)
                edge_index.to(self.device)
                #edge_attr.to(self.device)
                pc_ = point_cloud
                ed_ = edge_index 

                edge_index = edge_index.type(torch.int64)

                tool_obs = space["tool"]
                tool_obs.to(self.device)
            
                # set the edges to zero between the point cloud and the tool and one within the point cloud
                #edge_attributes_1 = torch.zeros((batch.edge_index.shape[1], 1), device=self.device)
                #edge_attributes_2 = torch.zeros((batch.edge_index.shape[1], 1), device=self.device)
                #edge_attributes_1[0:((4096*4)-1)] = edge_attr[0:((4096*4)-1)]
                #edge_attributes_2[4096:] = edge_attr[4096:]

                data_list_1 = [Data(x=point_cloud[i], edge_index=edge_index[i]) for i  in range(states.shape[0])] 
                loader_1 = DataLoader(data_list_1, batch_size = states.shape[0])
                batch_1 = next(iter(loader_1))

                
                # encode the point cloud
                x = self.pre_process(pc_)

                # set the edges attributes to zero between the point cloud and the tool and one within the point cloud
                x1 = x.view(-1, x.shape[-1])
                x = self.gat_conv_1(x = x1 , edge_index = batch_1.edge_index)
                x = F.leaky_relu(x, negative_slope=0.2)
                # set the edges to one between the point cloud and the tool and zero within the point cloud
                x = self.gat_conv_2(x = x, edge_index = batch_1.edge_index)
                x = F.leaky_relu(x, negative_slope=0.2)


                #extract the target and tool features from the point cloud
                # capture the target and tool features from the point cloud batch
                x = x.view(states.shape[0], 4098, -1)  # 4096 for the point cloud, 1 for the target, and 1 for the tool
                target_features = x[:, 4096, :]
                tool_features = x[:, 4097, :]
                # concatenate the target and tool features with the tool observations
                x = torch.cat([target_features, tool_features], dim=-1)
                
                x = self.post_process(x)

                gat_output = torch.cat([x.view(states.shape[0], -1),tool_obs.view(states.shape[0], -1)], dim = -1)

            else: 
                gat_output =self._gat_output
            self._gat_output = None
            output = self.value_layer(gat_output)
            output = output.to(self.device_output)
            return output, {}


