# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from lab_tasks.basetask import Basetask
from lab_tasks.baseline_zhang import BaselineZhang
from lab_tasks.baseline_soft import BaselineSoft
from lab_tasks.pulse import Pulse
from typing import Any, ClassVar
from isaacsim.core.api.loggers import DataLogger



from lab_tasks.basetask import Basetask

def get_task(cfg):
    if cfg["task"]["name"] == "baseline_zhang":
        task = BaselineZhang(cfg = cfg) 
    elif cfg["task"]["name"] == "baseline_soft":
        task = BaselineSoft(cfg = cfg)
    elif cfg["task"]["name"] == "gatpush":
        task = Pulse(cfg = cfg) 
    else: 
        task = BaselineZhang(cfg=cfg)
    return task


class BaseEnv(DirectRLEnv):
    def __init__(self, cfg_env, cfg_task):
        self.episode_idx = 0
        self.cfg_env= cfg_env
        self.cfg_task = cfg_task
        self.total_reward = 0.0
        self.total_timesteps = 0

        self.saving_path = self.setup_saving_path()

        import isaaclab.sim as sim_utils
        from isaaclab.assets import Articulation, ArticulationCfg
        from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.sim import SimulationCfg, RenderCfg
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
        from isaaclab.utils import configclass
        from isaaclab.utils.math import sample_uniform
        from isaaclab.actuators import ImplicitActuatorCfg
                # setup the main stage
        import isaacsim.core.utils.stage as stage_utils
        #self.num_envs=self.cfg_env["scene"]["num_envs"]
        self._device = self.cfg_env["env_cfg"]["device"]
        scene_usd_path = self.cfg_env["env_cfg"]["env_path"]
        stage_utils.add_reference_to_stage(usd_path=scene_usd_path, prim_path = "/World")
        if self.cfg_env["env_cfg"]["feature_extractor"]["type"] == "CNN": 
            self.spaces = {
            "fmap": spaces.Box(low = 0.0, high = 255.0, shape = (self.cfg_env["env_cfg"]["vision_h"],self.cfg_env["env_cfg"]["vision_w"],1), dtype= np.float32),
            "tool": spaces.Box(low = -float("inf"), high = float("inf"), shape = (self.cfg_env["env_cfg"]["tool_num_obs"],), dtype= np.float32),
            }
        elif self.cfg_env["env_cfg"]["feature_extractor"]["type"] == "GAT":
            edge_num = self.cfg_env["env_cfg"]["feature_extractor"]["shape"] * self.cfg_env["env_cfg"]["feature_extractor"]["k_neighbour"]
            for i in range(self.cfg_env["env_cfg"]["feature_extractor"]["virtual_node_link"] + 1):
                if i == 0:
                    edge_num += 0
                else:
                    edge_num += ((self.cfg_env["env_cfg"]["feature_extractor"]["shape"]) + (i - 1) ) * 2                                                         
            self.spaces = {
            "point_cloud": spaces.Box(low = -float("inf"), high = float("inf"), shape = (self.cfg_env["env_cfg"]["feature_extractor"]["shape"] + 2,3), dtype= np.float32),
            "edge_index": spaces.Box(low = -float("inf"), high = float("inf"), shape = (2, edge_num), dtype= np.int32),
            "edge_attribute": spaces.Box(low = -float("inf"), high = float("inf"), shape = (edge_num, 1), dtype= np.float32),
            "tool": spaces.Box(low = -float("inf"), high = float("inf"), shape = (self.cfg_env["env_cfg"]["tool_num_obs"],), dtype= np.float32),
            }
        elif self.cfg_env["env_cfg"]["feature_extractor"]["type"] == "None": 
            self.spaces = {
                "tool": spaces.Box(low = -float("inf"), high = float("inf"), shape = (self.cfg_env["env_cfg"]["tool_num_obs"],), dtype= np.float32),
            }

        self.robot_cfg = ArticulationCfg(
                prim_path = "/World/kinova_7dof_brush/kinova_7dof_brush/robot", 
                actuators={
                    "all_joints": ImplicitActuatorCfg(
                        joint_names_expr=[self.cfg_env["robot_cfg"]["joint_names_expr"]],
                        stiffness=self.cfg_env["robot_cfg"]["stiffness"],
                        damping=self.cfg_env["robot_cfg"]["damping"],
                    ),
                    },
            )
        class MySceneCfg(InteractiveSceneCfg):
            robot =  self.robot_cfg

        base_env_cfg = DirectRLEnvCfg(
                # env
            seed = self.cfg_env["env_cfg"]["seed"],
            decimation = self.cfg_env["env_cfg"]["decimation"],
            decimation_cmd = self.cfg_env["env_cfg"]["decimation_cmd"],
            decimation_ctrl = self.cfg_env["env_cfg"]["decimation_ctrl"],
            episode_length_s = self.cfg_env["env_cfg"]["episode_length_s"],
            action_space = spaces.Box(
                                    low = -1.0,
                                    high = 1.0,
                                    shape = (self.cfg_env["env_cfg"]["num_actions"],),
                                    dtype= np.float32,
                            ),
            observation_space = spaces.Dict(
                spaces = self.spaces,
            ),
            #observation_space = spaces.Box(low = -float("inf"), high = float("inf"), shape = (self.cfg_env["env_cfg"]["tool_num_obs"],), dtype= np.float32),

            state_space = 0,
            # simulation
            sim = SimulationCfg(
                dt=1 / self.cfg_env["simulation_cfg"]["freq"], 
                render_interval=self.cfg_env["env_cfg"]["decimation"],
                use_fabric = self.cfg_env["simulation_cfg"]["use_fabric"], 
                device = self._device, 
                render = RenderCfg(
                    enable_reflections = True, 
                    antialiasing_mode = "DLAA",
                    enable_dlssg = True, 
                    enable_dl_denoiser = True,
                    dlss_mode = 0,
                    enable_shadows = True,

                )
                ),
            # scene
            scene = MySceneCfg(num_envs=self.cfg_env["scene"]["num_envs"], env_spacing=0.0, replicate_physics=True),

        )
        from omni.physx import acquire_physx_interface
        physx_interface = acquire_physx_interface()
        physx_interface.overwrite_gpu_setting(1)
        self.task = self.setup_task(self.cfg_task)

        
        render_mode = 0
        kwargs = {} 
        super().__init__(base_env_cfg, render_mode, **kwargs)
        

    def setup_task(self, cfg):
        task = get_task(cfg)
        return task

    def _setup_scene(self):
        from isaaclab.scene import InteractiveSceneCfg
        from isaaclab.utils import configclass
        from isaaclab.scene import InteractiveScene
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg


        from isaaclab.assets import ArticulationCfg, Articulation
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.controllers import OperationalSpaceController, OperationalSpaceControllerCfg
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.sensors.camera import Camera, CameraCfg
        self.set_sim_to_task()
        # setup the articulation
        self.robot = self.scene.articulations["robot"]
        if self.cfg_env["controller"]["type"] == "task_space":
            # Task Space Controller
            controller_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
            controller = DifferentialIKController(controller_cfg, num_envs=self.scene.num_envs, device = self.cfg_task["task"]["device"])
        elif self.cfg_env["controller"]["type"] == "operational_space":
            controller_cfg = OperationalSpaceControllerCfg (
                            target_types=["pose_abs", "wrench_abs"],
                            impedance_mode="variable_kp",
                            inertial_dynamics_decoupling=True,
                            partial_inertial_dynamics_decoupling=False,
                            gravity_compensation=False,
                            motion_damping_ratio_task=1.0,
                            contact_wrench_stiffness_task=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
                            motion_control_axes_task=[1, 1, 0, 1, 1, 1],
                            contact_wrench_control_axes_task=[0, 0, 1, 0, 0, 0],
                            nullspace_control="position",
            )
            controller = OperationalSpaceController(controller_cfg, num_envs=self.scene.num_envs, device = self.cfg_task["task"]["device"])
        # setup the cameras
        camera_cfgs = []
        cameras = []
        for i in range(self.cfg_env["cameras"]["num"]):
            camera_cfgs.append(CameraCfg(
                prim_path=self.cfg_env["cameras"]["prim_path_" + str(i)],
                update_period= self.cfg_env["cameras"]["update_period"],
                height= self.cfg_env["cameras"]["height"],
                width=self.cfg_env["cameras"]["width"],
                data_types=self.cfg_env["cameras"]["data_types"],
            colorize_semantic_segmentation=self.cfg_env["cameras"]["colorize_semantic_segmentation"],
            colorize_instance_id_segmentation=self.cfg_env["cameras"]["colorize_instance_id_segmentation"],
            colorize_instance_segmentation=self.cfg_env["cameras"]["colorize_instance_segmentation"],
            spawn= None
            )
            )
            cameras.append(Camera(cfg=camera_cfgs[i]))
        # setup the datalogger
        if self.cfg_env["env_cfg"]["datalogger"] == True:
            self.datalogger = DataLogger()
            self.task.set_datalogger(self.datalogger)
            
        self.task.set_up_scene(device = self._device ,sim= self.sim, scene = self.scene, ik_controller = controller, cameras = cameras)

    def set_sim_to_task(self): 
        self.sim.get_physics_context().enable_gpu_dynamics(self.cfg_env["simulation_cfg"]["enable_gpu_dynamics"])
        self.sim.get_physics_context().set_solver_type(self.cfg_env["simulation_cfg"]["solver_type"])
        self.task.set_sim(self.sim)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if torch.any(torch.isnan(actions)) == True:
            print("observation is Nan")
            print("Observed fault: " + str(actions))
        if torch.any(torch.isinf(actions)) == True:
            print("observation is Inf")
            print("Observed fault: " + str(actions))
        obs = torch.nan_to_num(actions)
        actions[:] = torch.tensor([self.cfg_env["env_cfg"]["action_scale"]], device = self._device)[:] * actions.clone()[:]
        self.task.process_actions(actions)

    def _apply_action(self) -> None:
        self.task.apply_action()
        if self.cfg_env["env_cfg"]["datalogger"] == True:
            logging_data = self.task.get_logging_data()
            #print("Logging data: " + str(logging_data))
            self.datalogger.add_data(
                data = logging_data,
                current_time_step = self.total_timesteps - 1,
                current_time = self.sim.current_time,
            )
            self.datalogger.start()

    def _get_observations(self) -> dict:
        self.total_timesteps = self.total_timesteps + 1
        observations = self.task.get_observations()
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward = self.task.compute_rewards()
        return reward    

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        success = self.task.is_done()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if success or time_out:
            print("Total reward: " + str(self.total_reward))
            print("Total timesteps: " + str(self.total_timesteps))
            if self.cfg_env["env_cfg"]["datalogger"] == True:
                saving_path = self.saving_path + "/episode_" + str(self.episode_idx) + ".json"
                self.datalogger.save(saving_path)
                self.total_timesteps = 0
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self.task.reset(env_ids)
        self.episode_idx += 1
        # reset the login buffer data
        if self.cfg_env["env_cfg"]["datalogger"] == True:
            self.datalogger.reset()

    def _update_obj_pos(self):
        self.task.update_obj_pos()

    def setup_saving_path(self):
        import os
        base_dir = self.cfg_env["env_cfg"]["datalog_path"]
        algo_type = self.cfg_env["env_cfg"]["type"]
        iteration  = 1
        saving_path = base_dir + "/play_" + algo_type + "_" + str(iteration)
        while os.path.exists(saving_path):
            iteration += 1
            saving_path = base_dir + "/play_" + algo_type + "_" + str(iteration)
        os.makedirs(saving_path, exist_ok=True)
        return saving_path
    






        

