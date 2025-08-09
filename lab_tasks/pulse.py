from __future__ import annotations
import torch as th
from collections.abc import Sequence
from lab_tasks.basetask import Basetask
import numpy as np
import time
import matplotlib.pyplot as plt
from utils.mapping import Mapping
import open3d as o3d
import open3d.core as o3c
import torchvision.transforms as Timg
import torchvision
import torch.nn.functional as F
import warp as wp
from utils.object_manager import ObjectManager
from utils.vision.dvision_graph import Dvision
from utils.admittance_controller import AdmittanceController


class Pulse(Basetask):
    def __init__(self, cfg):
        #import utilities
        self.cfg = cfg
        self.cpt = 0
        self.num_envs = self.cfg["task"]["num_envs"]
        #self.particle_prim, self.initial_particle_pos= self.get_particles_position()
        self.good_reach = False
        self.obj_prim_path = "/World/object"
        self.episode_cpt = 0
        self.bad_episode = False
        self.sim_prev_time = 0


    def initialise_image_array(self):
        self.full_data_array  = []
    
    def write_image_to_container(self, image):
        self.full_data_array.append(image)
        return
    
    def save_numpy_data(self, array, episode_index):
        filename = self.cfg["task"]["recording"]["file_path"] + "dataset_" + str(episode_index) + ".npy"
        np.save(filename, np.array(array))


    def set_up_scene(self, device , sim, scene, ik_controller, cameras): 
        from isaacsim.core.prims import Articulation, SingleArticulation
        if self.cfg["task"]["play"] == True: 
            if self.cfg["task"]["recording"]["enabled"] == True: 
                self.initialise_image_array()
        self.device = self.cfg["task"]["device"]
        self.sim_device = device
        self.obj_manager = ObjectManager(
                        cfg = self.cfg["task"]["objects"],
                        obj_prim_path = self.obj_prim_path,
                        device = self.device
                        )
        self.sim = sim
        self.scene = scene
        self.robot = self.scene.articulations["robot"]
        self.ik_controller = ik_controller
        self.cameras = cameras
        self.visualiser = self.set_target_marker()
        self.prev_time = time.time() 
        self.sim_prev_time = self.sim.current_time
        self.ee_prim = self.get_ee_prim()
        self.obj_prim = self.get_object_prim()
        self.particule_vis = self.set_part_marker()
        self.particule_middle_vis = self.set_part_marker_middle()
        self.particles_target_f_prev = 0
        self.particles_target_dist_mean_prev = 0
        self.particles_ee_dist_prev = 0
        self.ee_fpart_pos_prev = 0
        self.part_target_pos_prev = 0
        self.ee_part_pos_prev = th.zeros((1,3), device = self.device ) 
        # import depth vision module
        self.dvision = Dvision(self.cfg)
        self.dvision.load_vae_model()
        self.articulation_prim = Articulation(prim_paths_expr = "/World/kinova_7dof_brush/kinova_7dof_brush/robot")
        



    def set_target_marker(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        # set up the visual marker (for indication only)
        marker_cfg = VisualizationMarkersCfg(
            prim_path = self.cfg["task"]["target"]["prim_path"],
            markers={
                "target": sim_utils.CylinderCfg(
                radius= self.cfg["task"]["target"]["radius"],
                height= self.cfg["task"]["target"]["height"],
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                )
            },
        )
        visualiser = VisualizationMarkers(marker_cfg)
        return visualiser
    
    def set_part_marker(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        # set up the visual marker (for indication only)
        marker_cfg = VisualizationMarkersCfg(
            prim_path = self.cfg["task"]["target"]["prim_path"],
            markers={
                "particle_f": sim_utils.CuboidCfg(
                    size=(0.01, 0.01, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 1.0)),
                )
            },
        )
        visualiser = VisualizationMarkers(marker_cfg)
        return visualiser
    
    def set_part_marker_middle(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        # set up the visual marker (for indication only)
        marker_cfg = VisualizationMarkersCfg(
            prim_path = self.cfg["task"]["target"]["prim_path"],
            markers={
                "particle_f": sim_utils.CuboidCfg(
                    size=(0.01, 0.01, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                )
            },
        )
        visualiser = VisualizationMarkers(marker_cfg)
        return visualiser
    
    def get_ee_prim(self):
        from isaacsim.core.prims import XFormPrim
        ee_prim = XFormPrim(
            prim_paths_expr = self.cfg["task"]["tool"]["prim_path_bristles"] ,
            name = "ee",
        )
        return ee_prim
    
    def get_object_prim(self):
        self.obj_manager.randomize_properties()
        prim = self.obj_manager.redefine_object()
        
        return prim
        
    
    def get_observations(self):
        from isaaclab.utils.math import euler_xyz_from_quat,make_pose,matrix_from_quat,pose_inv,unmake_pose, unproject_depth, transform_points
        
        from torch_geometric.data import Data
        import torch_geometric.transforms as T
        from torch_geometric.transforms import SamplePoints, KNNGraph, GridSampling
        #from isaaclab.utils.math import transform_points, unproject_depth
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
        import torch.utils.dlpack
        import isaacsim.core.utils.bounds as bounds_utils

        # read real time for fps estimation, not used for control
        _time = time.time() 
        Dt = _time - self.prev_time
        self.prev_time = _time
        #print("frame number: " + str(1/Dt))

        # read simulation time for control
        _sim_time = self.sim.current_time
        self.sim_dt = _sim_time - self.sim_prev_time
        self.sim_prev_time = _sim_time


  
        # tool observations
        #print("usd ee position: " + str(self.ee_prim.get_world_poses()))

        self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        self.joint_pos = self.robot.data.joint_pos.clone().to(self.device)
        ee_pose = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7].clone()
        ee_vels = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 7:13].clone()
        ee_position = ee_pose[:,0:2].to(self.device)
        #print("lab ee position: " + str(self.ee_position))
        self.ee_orientation_rad = euler_xyz_from_quat(ee_pose[:,3:7])
        self.ee_orientation_rad = th.tensor([self.ee_orientation_rad[0], self.ee_orientation_rad[1], self.ee_orientation_rad[2]], device=self.device)
        self.ee_velocity = ee_vels[:,0:2].to(self.device)
        self.ee_angular_velocity = ee_vels[:,3:6].to(self.device)
        #ee_pos_x = np.random.uniform(low= 0.1, high= 1.0)
        #ee_pos_y = np.random.uniform(low= -0.4, high = 0.4)
        #ee_pos_z = 0.14
        #ree_pos = th.tensor([[ee_pos_x, ee_pos_y]], device = self.device)

        #obj_pos_x = np.random.uniform(low= 0.1, high= 1.1)
        #obj_pos_y = np.random.uniform(low= -0.4, high = 0.4)
        #obj_pos_z = 0
        
        #obs_pos = th.tensor([[obj_pos_x, obj_pos_y, obj_pos_z]], device = self.device)

        #obj_ori_z = np.random.uniform(low = 0, high = 2 * np.pi)

        #self.obj_manager.redefine_object(obs_pos, obj_ori_z)

        medium = self.dvision.process_point_cloud(self.cameras, self.sim.get_physics_dt())
        if th.numel(medium) == 0:
            print("there is no points in the point cloud")
            self.bad_episode = True
        structured_graph = self.dvision.update_point_cloud(
            medium_tensor_pc= medium, 
            target_pc = self.target_lists,
            ee_position = ee_pose[:,0:3].to(self.device),
            ee_orientation = self.ee_orientation_rad,
            Dt = Dt
        )

        # cartesian and absolutes distances
        dist_pos = self.dvision.calculate_distances(
            medium_tensor_pc = medium, 
            target_position=self.target_position,
            ee_position = ee_position, 
            medium_visualiser = self.particule_middle_vis,
            farthest_visualiser = self.particule_vis ,
        )
        
        self.ee_particle_pos = dist_pos["ee_particle_pos"]
        self.ee_particle_dist = dist_pos["ee_particle_dist"]
        self.particle_target_pos = dist_pos["particle_target_pos"]
        self.particle_target_dist = dist_pos["particle_target_dist"]
        self.part_pos = dist_pos["particle_mean_pos"]
        self.ee_fpart_pos = dist_pos["ee_fpart_pos"]
        self.ee_fpart_dist = dist_pos["ee_fpart_dist"]


        if th.any(th.isnan(self.ee_particle_pos)) == True:
            print("ee_particle_pos is Nan")
            print("Observed fault: " + str(self.ee_particle_pos))
            self.bad_episode = True
        if th.any(th.isinf(self.ee_particle_pos)) == True:
            print("ee_particle_pos is Inf")
            print("Observed fault: " + str(self.ee_particle_pos))
        self.ee_particle_pos = th.nan_to_num(self.ee_particle_pos)

        if th.any(th.isnan(self.particle_target_pos)) == True:
            print("particle_target_pos is Nan")
            print("Observed fault: " + str(self.particle_target_pos))
            self.bad_episode = True
        if th.any(th.isinf(self.particle_target_pos)) == True:
            print("particle_target_pos is Inf")
            print("Observed fault: " + str(self.particle_target_pos))
        self.particle_target_pos = th.nan_to_num(self.particle_target_pos)

        #print("latent space shape : " + str(depth_latent_space.shape))
        #print("ee part pos: " + str(m_part_pos))
        tool_obs =  th.concatenate(
                [
                    self.ee_orientation_rad,
                    #self.ee_particle_pos.squeeze(dim=0), # dim 3ff
                    #self.particle_target_pos.squeeze(dim=0),
                ])
        
        if th.any(th.isnan(tool_obs)) == True:
            print("particle_target_pos is Nan")
            print("Observed fault: " + str(tool_obs))
            self.bad_episode = True
        if th.any(th.isinf(tool_obs)) == True:
            print("particle_target_pos is Inf")
            print("Observed fault: " + str(tool_obs))
            self.bad_episode = True
        tool_obs = th.nan_to_num(tool_obs)
        obs = {
            "point_cloud": structured_graph.x,
            "edge_index": structured_graph.edge_index,
            "edge_attribute": structured_graph.edge_attr,
            "tool": tool_obs,
        }
        return {"policy": obs}



        

    def reset(self, env_ids: Sequence[int] | None):
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils.math import euler_xyz_from_quat
        self.episode_cpt = self.episode_cpt + 1
        self.bad_episode = False
        self.robot.reset()
        
        self.ik_controller.reset()

        #if self.cfg["task"]["play"] == True: 
        #    if self.cfg["task"]["recording"]["enabled"] == True: 
        #        self.save_numpy_data( self.full_data_array, self.episode_cpt)
        #        self.initialise_image_array()
       # # particle position 

        self.good_reach = False


        #from isaaclab.managers import SceneEntityCfgs
        self.init_cmd = self.reset_robot_position()
        self.ik_controller.set_command(self.init_cmd)

        # target reset
        self.target_position = self.reset_target_position()
        self.target_lists = self.dvision.generate_target_points(self.target_position, init_radius = 0.02)
        
        self.obj_manager.remove_object_from_work_table(self.obj_prim)
        self.ee_jacobi_idx = 7
        joint_ids = [0, 1, 2, 3, 4, 5, 6]
        body_ids = 8
        

        joints_command = th.tensor([0,-(np.pi/2),0,(np.pi/2),0,(np.pi/2),0], device = "cpu") #self.device)
        self.articulation_prim.set_joint_positions(joints_command) # joint_indices=joint_ids)
        self.sim.step()
        self.scene.update(1/self.cfg["task"]["freq"])

        for i in range(self.cfg["task"]["init"]["delay"]):
            ee_pose = self.robot.data.body_state_w[:, body_ids, 0:7].clone().to(self.device)
            jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, joint_ids].to(self.device)
            joint_pos = self.robot.data.joint_pos.clone().to(self.device)
            joints_command = self.ik_controller.compute(ee_pose[:, 0:3], ee_pose[:, 3:7],jacobian,joint_pos ).to(self.sim_device)
            self.articulation_prim.set_joint_positions(joints_command) 
            self.sim.step()
            self.scene.update(1/self.cfg["task"]["freq"])
        
        self.obj_manager.randomize_properties()
        self.reset_rigid_object_position(self.obj_prim, self.init_cmd[:,0:2], self.target_position[:,0:2])
        self.obj_manager.randomize_surface_friction()
        self.sim.step()
        self.scene.update(1/self.cfg["task"]["freq"])

        

        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["joint_.*"], body_names=["end_effector_link"])
        self.robot_entity_cfg.resolve(self.scene)

        # read robot state
        ee_pose = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7].clone().to(self.device)
        ee_position = ee_pose[:,0:2]
        #self.ee_ftarget_dist_init = th.norm(ee_position - self.fixed_target)
        #ee_orientation_rad = euler_xyz_from_quat(ee_pose[:,3:7])
        #ee_orientation_rad = th.tensor([ee_orientation_rad[0], ee_orientation_rad[1], ee_orientation_rad[2]], device=self.device)
        
        self.robot.reset()
        self.ik_controller.reset()

        medium = self.dvision.process_point_cloud(self.cameras, self.sim.get_physics_dt())
        # cartesian and absolutes distances
        dist_pos = self.dvision.calculate_distances(
            medium_tensor_pc = medium, 
            target_position=self.target_position,
            ee_position = ee_position, 
            medium_visualiser = self.particule_middle_vis,
            farthest_visualiser = self.particule_vis,
        )
        self.ee_particle_dist_init = dist_pos["ee_particle_dist"]
        self.particle_target_dist_init = dist_pos["particle_target_dist"]
        self.ee_fpart_dist_init = dist_pos["ee_fpart_dist"]
        #print("f part pos : " + str(dist_pos["fpart_pos"]))
        self.prev_obj_pos = dist_pos["fpart_pos"][0:2]
        print("reset done")

    def is_done(self) -> tuple[th.Tensor, th.Tensor]:
        done = False
        done = done or self.target_reached() or self.bad_episode
        return done

    def process_actions(self, actions): 
        from isaaclab.utils.math import quat_from_euler_xyz
        from isaaclab.utils.math import euler_xyz_from_quat

        #_sim_time = self.sim.current_time
        #self.sim_dt = _sim_time - self.sim_prev_time
        #self.sim_prev_time = _sim_time

        # get the current ee state
        act = actions.clone()
        if act.shape[0] > 1: 
            act = act.unsqueeze(dim=0)
        Pxy = act[:,0:2] 
        Dpsi = act[:,[2]]


        ee_pose = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7].clone().to(self.device)

        ang_command = th.Tensor([np.pi, 0, np.pi]).to(self.device)
        ee_cmd_orientation_rad = euler_xyz_from_quat(ee_pose[:,3:7])

        ang_command[2] = ee_cmd_orientation_rad[2] + th.tensor((Dpsi * (np.pi/180)), device = self.device)
        ang_command[2] = ((ang_command[2] + np.pi) % (2*np.pi)) - np.pi

        #print("ang command: " + str(ang_command[2]))

        ang_command = quat_from_euler_xyz(ang_command[0], ang_command[1], ang_command[2])
        self.ang_command = th.tensor([[ang_command[0],ang_command[1], ang_command[2], ang_command[3]]]).to(self.device)
        
        self.P_rate = th.tensor([[(ee_pose[0][0] + Pxy[0][0]), (ee_pose[0][1] +  Pxy[0][1]), 0]], device = self.device)


    def update_obj_pos(self):
        pass

    def estimate_obj_velocity(self, dt, obj_pos):
        _dt = th.tensor([[dt, dt]], device = self.device)
        vel = (obj_pos - self.prev_obj_pos) / _dt
        self.prev_obj_pos = obj_pos
        return vel


    def apply_action(self):
        from isaacsim.core.utils.types import ArticulationActions
        from isaaclab.utils.math import quat_from_euler_xyz
        from isaaclab.utils.math import euler_xyz_from_quat

        # read simulation time for control
        ee_pose = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7].clone().to(self.device)

        pos_command = self.P_rate
        pos_command[:,2] = self.cfg["task"]["action"]["z_command"]

        if pos_command[:,0] < self.cfg["task"]["action"]["xmin_limit"]:
            pos_command[:,0] = self.cfg["task"]["action"]["xmin_limit"]
        if pos_command[:,0] > self.cfg["task"]["action"]["xmax_limit"]:
            pos_command[:,0] = self.cfg["task"]["action"]["xmax_limit"]
        if pos_command[:,1] < self.cfg["task"]["action"]["ymin_limit"]:
            pos_command[:,1] = self.cfg["task"]["action"]["ymin_limit"]
        if pos_command[:,1] > self.cfg["task"]["action"]["ymax_limit"]:
            pos_command[:,1] = self.cfg["task"]["action"]["ymax_limit"]

        command = th.tensor([[pos_command[:,0], pos_command[:,1], pos_command[:,2], self.ang_command[:,0], self.ang_command[:,1], self.ang_command[:,2], self.ang_command[:,3]]])
        self.ik_controller.set_command(command)

        jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids].clone().to(self.device)      
        ee_pose = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7].clone().to(self.device)
        joint_pos = self.robot.data.joint_pos.clone().to(self.device)

        # update command to joint space
        joints_command = self.ik_controller.compute(ee_pose[:, 0:3], ee_pose[:, 3:7],jacobian,joint_pos ).to(self.sim_device)  
        _pi = th.tensor([np.pi], device = self.sim_device)
        _2_pi = th.tensor([2*np.pi], device = self.sim_device)
        joints_command[:,6] = th.remainder((joints_command[:,6] + _pi), _2_pi) - _pi
        self.robot.set_joint_position_target(joints_command, joint_ids=self.robot_entity_cfg.joint_ids)
        

    def compute_rewards(self)-> th.Tensor:
        reward = th.zeros((1)).to(self.device)
        dist_a = 0
        dist_b = 0
        dist_c = 0


        #dist_a = - self.dist_normalized(0, self.ee_fpart_dist_init  , self.ee_fpart_dist)
        dist_b =  - self.dist_normalized(0, self.ee_particle_dist_init, self.ee_particle_dist)
        dist_c = - self.dist_normalized(0, self.particle_target_dist_init, self.particle_target_dist)
        #dist_d = - self.dist_normalized(0, self.ee_ftarget_dist_init, self.ee_ftarget_dist)
        reward[0] =  dist_b + dist_c 
        #print("dist a reward: " + str(dist_a))
        #print("dist b reward: " + str(dist_b))
        #print("dist c reward: " + str(dist_c))
        
        if self.good_reach == True:
            reward[0] = 50
        #print("reward : " + str(reward))

        return reward.to("cpu")
    
    def dist_normalized(self, c_min, c_max, dist, what="object"):
        normed_min, normed_max = 0, 1
        if (dist - c_min) == 0 or (c_max - c_min) == 0:
            print("dist: ", dist, "c_min: ", c_min, "c_max: ", c_max, "for: ", what)
        x_normed = (dist - c_min) / (c_max - c_min)
        x_normed = x_normed * (normed_max - normed_min) + normed_min
        return th.round(x_normed, decimals = 4)

    
    def reset_robot_position(self):
        from isaaclab.utils.math import quat_from_euler_xyz
        ee_init_pos = self.cfg["task"]["init"]["robot"]
        dx = self.cfg["task"]["init"]["robot_dx"]
        dy = self.cfg["task"]["init"]["robot_dy"]
        robot_x = np.random.uniform(low= ee_init_pos[0] - dx, high= ee_init_pos[0] + dx)
        robot_y = np.random.uniform(low= ee_init_pos[1] - dy, high= ee_init_pos[1] + dy)
        robot_z = self.cfg["task"]["action"]["z_command"]
        robot_x = np.round(robot_x, decimals = 4)
        robot_y = np.round(robot_y, decimals = 4)
        yaw = np.random.uniform(low=(0), high = 2 * (np.pi)) 

        pos_command = th.tensor([[robot_x, robot_y, robot_z]])
        
        quat_command = quat_from_euler_xyz(th.tensor([np.pi]), th.tensor([0]), th.tensor([yaw]))
        quat_command = th.tensor([[quat_command[:,0],quat_command[:,1], quat_command[:,2], quat_command[:,3]]])
        command = th.tensor([[pos_command[:,0], pos_command[:,1], pos_command[:,2], quat_command[:,0], quat_command[:,1], quat_command[:,2], quat_command[:,3]]], device = self.device)
        return command
    
    def reset_target_position(self):
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        import isaaclab.sim as sim_utils
        from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
        target_init_pos = self.cfg["task"]["init"]["target"]
        dx = self.cfg["task"]["init"]["target_dx"]
        dy = self.cfg["task"]["init"]["target_dy"]
        target_x = np.random.uniform(low= target_init_pos[0] - dx, high= target_init_pos[0] + dx)
        target_y = np.random.uniform(low= target_init_pos[1] - dy, high= target_init_pos[1] + dy) 
        target_position = th.tensor([[target_x, target_y, 0.01]], device = self.device)

        self.visualiser.visualize(target_position)

        return target_position
    
    def reset_rigid_object_position(self, prim, ee_pos, target_pos):
        init_pos = np.array(self.cfg["task"]["init"]["medium"])
        dx = self.cfg["task"]["init"]["debris_dx"]
        dy = self.cfg["task"]["init"]["debris_dy"]
        medium_x = np.random.uniform(low= init_pos[0]-dx, high=init_pos[0]+dx)
        medium_y = np.random.uniform(low= init_pos[1]-dy, high=init_pos[1]+dy) 

        medium_pos = th.tensor([[medium_x, medium_y, init_pos[2]]], device = self.device)

        dist_a = th.norm(medium_pos[:,0:2] - ee_pos).item()
        dist_b = th.norm(medium_pos[:,0:2] - target_pos).item()

        while(dist_a <=  self.cfg["task"]["init"]["object_ee_min_dist"] or dist_b <=  self.cfg["task"]["init"]["debris_target_min_dist"]):
            medium_x = np.random.uniform(low= init_pos[0]-dx, high=init_pos[0]+dx)
            medium_y = np.random.uniform(low= init_pos[1]-dy, high=init_pos[1]+dy) 
            medium_pos = th.tensor([[medium_x, medium_y, init_pos[2]]], device = self.device)
            dist_a = th.norm(medium_pos[:,0:2] - ee_pos).item()
            dist_b = th.norm(medium_pos[:,0:2] - target_pos).item()
        
        obj_ori_z = np.random.uniform(low = 0, high = 2 * np.pi)

        self.obj_prim = self.obj_manager.redefine_object(medium_pos, heading = obj_ori_z)

    def reset_particles_position(self, prim, initial_position, ee_pos, target_pos):
        from pxr import Usd, UsdGeom, Gf
        import copy
        particleSet_init_pos = self.cfg["task"]["particles"]["position"]
        dx = self.cfg["task"]["init"]["debris_dx"]
        dy = self.cfg["task"]["init"]["debris_dy"]
        medium_x = np.random.uniform(low= particleSet_init_pos[0]-dx, high=particleSet_init_pos[0]+dx)
        medium_y = np.random.uniform(low= particleSet_init_pos[1]-dy, high=particleSet_init_pos[1]+dy) 

        particles_pos_tensor = th.tensor(initial_position, device = self.device)
        particles_pos_x = particles_pos_tensor[:,[0]].flatten()
        particles_pos_y = particles_pos_tensor[:,[1]].flatten()
        particles_pos_x_mean = th.mean(particles_pos_x) + medium_x
        particles_pos_y_mean = th.mean(particles_pos_y) + medium_y
        particles_pos_mean = th.tensor([particles_pos_x_mean, particles_pos_y_mean], device = self.device)

        dist_a = th.norm(particles_pos_mean - ee_pos).item()
        dist_b = th.norm(particles_pos_mean - target_pos).item()
        while(dist_a <=  self.cfg["task"]["init"]["debris_target_min_dist"] or dist_b <=  self.cfg["task"]["init"]["debris_target_min_dist"]):
            medium_x = np.random.uniform(low= particleSet_init_pos[0]-dx, high=particleSet_init_pos[0]+dx)
            medium_y = np.random.uniform(low= particleSet_init_pos[1]-dy, high=particleSet_init_pos[1]+dy) 
            particles_pos_x_mean = th.mean(particles_pos_x) + medium_x
            particles_pos_y_mean = th.mean(particles_pos_y) + medium_y
            particles_pos_mean = th.tensor([particles_pos_x_mean, particles_pos_y_mean], device = self.device)
            dist_a = th.norm(particles_pos_mean - ee_pos).item()
            dist_b = th.norm(particles_pos_mean - target_pos).item()

        new_points = th.tensor(initial_position, device = self.device)
        new_points = new_points + th.tensor([medium_x, medium_y, 0], device = self.device)

        UsdGeom.Points(prim).GetPointsAttr().Set(new_points.to("cpu").numpy())
        


    def get_particles_position(self):
        import omni.kit.app
        from pxr import Usd, UsdGeom, Gf
        import omni.usd
        self.stage = omni.usd.get_context().get_stage()
        prim = self.stage.GetPrimAtPath("/World/ParticleSet")
        # Gets particles' positions and velocities 
        particles_pos = UsdGeom.Points(prim).GetPointsAttr().Get()
        return prim, particles_pos
    
    def target_reached(self): 
        reach = False
        if self.particle_target_dist < self.cfg["task"]["done"]["debris_target_mean_dist"]:
            reach = True
            self.good_reach = True
            print("---- Target Reached ----")
        else: 
            reach = False
        return reach
    


 
    



