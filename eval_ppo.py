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


from torch.optim.lr_scheduler import StepLR, LinearLR

#from models.PPO.ppo import PPO_Policy, PPO_Value
from models.PPO.ppo_cnn_shared import Shared_PPO_CNN
from models.PPO.ppo_gat_shared import Shared_PPO_GAT
from models.utils.CustomStatePrepocessor import CustomStatePreprocessor

import wandb

#from callback import GetDataCallback
import torch as th
th.autograd.set_detect_anomaly(True)
import numpy as np
np.seterr(all='raise')
import argparse
from typing import Callable
import datetime
import os
import shutil
import sys
import hydra
import gymnasium
from omegaconf import DictConfig, OmegaConf
from app.app import App

import logging

logging.getLogger("requests").setLevel(logging.WARNING)


# seed for reproducibility
set_seed(42)  # e.g. `set_seed(42)` for fixed seed



def setup_training_configuration(cfg, env, wconf) -> None:

    print("Selected Policy Type: " + str(cfg["RL"]["policy"]))
    print("Selected algorithm Type: " + str(cfg["RL"]["algo"]["type"]))

    if cfg["RL"]["algo"]["type"] == "PPO": 

        device = cfg["RL"]["algo"]["device"]
        device_model = cfg["RL"]["algo"]["device_model"]
        activation = cfg["RL"]["algo"]["activation"]
        value_arch = cfg["RL"]["algo"]["arch"]
        policy_arch = cfg["RL"]["algo"]["arch"]
        models = {}
        #models["policy"] = PPO_Policy(env.observation_space, env.action_space, device, clip_actions=True, num_envs=env.num_envs, arch=policy_arch, activation=activation) #, vision_w=cfg["RL"]["vision_h"], vision_h=cfg["RL"]["vision_w"], num_robot_obs=cfg["RL"]["tool_num_obs"])
        #models["value"] = PPO_Value(env.observation_space, env.action_space, device, num_envs=env.num_envs, arch=value_arch, activation=activation) #, vision_w=cfg["RL"]["vision_h"], vision_h=cfg["RL"]["vision_w"], num_robot_obs=cfg["RL"]["tool_num_obs"])

        if cfg["RL"]["algo"]["feature_extractor"] == "CNN":
            models["policy"] = Shared_PPO_CNN(env.observation_space, env.action_space, device, clip_actions=True, num_envs=1, arch=policy_arch, activation=activation, vision_w=cfg["RL"]["vision_h"], vision_h=cfg["RL"]["vision_w"], num_robot_obs=cfg["RL"]["tool_num_obs"])
            models["value"] = models["policy"]
        elif cfg["RL"]["algo"]["feature_extractor"] == "GAT":
            print("GAT policy selected")
            models["policy"] = Shared_PPO_GAT(env.observation_space, env.action_space, device_model, device, clip_actions=True, num_envs=1, arch=policy_arch, activation=activation, vision_w=cfg["RL"]["vision_h"], vision_h=cfg["RL"]["vision_w"], num_robot_obs=cfg["RL"]["tool_num_obs"])
            models["value"] = models["policy"]


        # instantiate a memory as rollout buffer (any memory can be used for this)
        memory = RandomMemory(memory_size=cfg["RL"]["algo"]["rollouts"], num_envs=env.num_envs, device=device)

        dcfg = PPO_DEFAULT_CONFIG.copy()
        dcfg["rollouts"] = cfg["RL"]["algo"]["rollouts"] #1024  # memory_size
        dcfg["learning_epochs"] = cfg["RL"]["algo"]["learning_epochs"]  #10
        batch_size = cfg["RL"]["algo"]["mini_batch_size"] 
        dcfg["mini_batches"] = int(cfg["RL"]["algo"]["rollouts"] / batch_size)
        dcfg["discount_factor"] =  cfg["RL"]["algo"]["discount_factor"]
        dcfg["lambda"] = cfg["RL"]["algo"]["lambda"]
        dcfg["learning_rate"] = cfg["RL"]["algo"]["learning_rate"]
        dcfg["learning_rate_scheduler"] = KLAdaptiveRL #StepLR #KLAdaptiveRL
        dcfg["learning_rate_scheduler_kwargs"] = {
            #"step_size": cfg["RL"]["algo"]["learning_epochs"],
            #"gamma": cfg["RL"]["algo"]["lr_gamma"]
            "kl_threshold": cfg["RL"]["algo"]["kl_threshold"], 
            "min_lr": cfg["RL"]["algo"]["min_lr"],
            "max_lr": cfg["RL"]["algo"]["max_lr"],
            } 
        dcfg["grad_norm_clip"] = cfg["RL"]["algo"]["grad_norm_clip"]
        dcfg["ratio_clip"] = cfg["RL"]["algo"]["ratio_clip"]
        dcfg["value_clip"] = cfg["RL"]["algo"]["value_clip"]
        dcfg["clip_predicted_values"] = cfg["RL"]["algo"]["clip_predicted_values"]
        dcfg["entropy_loss_scale"] = cfg["RL"]["algo"]["entropy_loss_scale"]
        dcfg["value_loss_scale"] = cfg["RL"]["algo"]["value_loss_scale"]
        dcfg["kl_threshold"] = cfg["RL"]["algo"]["kl_threshold_stop"]
        if cfg["RL"]["algo"]["state_preprocessor"] == "RunningStandardScaler":
            print("Using RunningStandardScaler as state preprocessor")
            dcfg["state_preprocessor"] = RunningStandardScaler
            dcfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
        elif cfg["RL"]["algo"]["state_preprocessor"] == "CustomStatePreprocessor":
            dcfg["state_preprocessor"] = CustomStatePreprocessor
            dcfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
        else: 
            dcfg["state_preprocessor"] = None
            dcfg["state_preprocessor_kwargs"] = {}
        if cfg["RL"]["algo"]["value_preprocessor"] == "RunningStandardScaler":
            print("Using RunningStandardScaler as state preprocessor")
            dcfg["value_preprocessor"] = RunningStandardScaler
            dcfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        else:
            dcfg["value_preprocessor"] = None
            dcfg["value_preprocessor_kwargs"] = {}



        # Logging to Weights and Biases (To do)
        dcfg["experiment"]["directory"] = cfg["RL"]["experiment"]["directory"]
        dcfg["experiment"]["experiment_name"] = cfg["RL"]["experiment"]["experiment_name"]
        dcfg["experiment"]["write_interval"] = cfg["RL"]["algo"]["rollouts"]
        dcfg["experiment"]["checkpoint_interval"] = cfg["RL"]["experiment"]["checkpoint_interval"]
        dcfg["experiment"]["store_separately"] = cfg["RL"]["experiment"]["store_separately"]
        dcfg["experiment"]["wandb"] = cfg["RL"]["experiment"]["wandb"]

        # Defining bespoke karg to allow sweeping 
        dcfg["experiment"]["wandb_kwargs"]["project"] = cfg["RL"]["experiment"]["wandb_kwargs"]["project"]
        dcfg["experiment"]["wandb_kwargs"]["mode"] = cfg["RL"]["experiment"]["wandb_kwargs"]["mode"]
        #dcfg["experiment"]["wandb_kwargs"]["config"] = wconf
        dcfg["experiment"]["wandb_kwargs"]["monitor_gym"] = cfg["RL"]["experiment"]["wandb_kwargs"]["monitor_gym"]
        dcfg["experiment"]["wandb_kwargs"]["sync_tensorboard"] = cfg["RL"]["experiment"]["wandb_kwargs"]["sync_tensorboard"]


        model = PPO(
                models = models, 
                memory = memory,
                cfg = dcfg, 
                observation_space = env.observation_space,
                action_space = env.action_space, 
                device = device
        )

        if cfg["RL"]["load_policy"] == True:
            print("Loading the policy from: " + str(cfg["RL"]["load_policy_path"]))
            model.load(cfg["RL"]["load_policy_path"])
            print("Policy loaded successfully")
        else:
            print("No policy loaded, using random policy for training")


    return model, models



@hydra.main(config_name="defaults", config_path="cfg")
def main(cfg: DictConfig):
    th.cuda.empty_cache()
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True

    )
    print("Config data: " + str(wandb.config))
        # create isaac environments

    app = App(wandb.config["app"])
    #sim_context = app.set_simulation_context()
    env_name = wandb.config["env"]["name"]
    from lab_envs.base_env import BaseEnv
    #from skrl.envs.loaders.torch import load_isaaclab_env
    env = gymnasium.make(id="BaseEnv-v0", cfg_env = wandb.config["env"], cfg_task  = wandb.config["tasks"])
    env = wrap_env(env, wrapper="isaaclab")

    model, gnn = setup_training_configuration(cfg=wandb.config["play"], 
                env = env,
                wconf= wandb.config,
                )
    if wandb.config["play"]["RL"]["algo"]["feature_extractor"] == "GAT":
        env.set_model(gnn["policy"])
    #wandb.config = cfg
    wandb.init(**cfg.wandb.setup, config=wandb.config, monitor_gym = True, save_code = True, sync_tensorboard = True)
    #wandb.run.log_code(".")
    #wandb.save(wandb.config.task["task"]["path"])
    

    # configure and instantiate the RL trainer
    cfg_trainer = {
                    "timesteps": wandb.config["play"]["RL"]["max_timestep"],
                    "headless": wandb.config["play"]["RL"]["headless"],
                    "disable_progressbar": wandb.config["play"]["RL"]["disable_progressbar"],
                    }
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[model])

    # start evaluation

    print("Evaluating the model")
    trainer.eval()


    app.close()



    


if __name__ == "__main__":
    main()
