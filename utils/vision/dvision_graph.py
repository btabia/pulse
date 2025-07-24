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
from  utils.vision.models.coordvae import CoordVAE

class Dvision: 
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = self.cfg["task"]["device"]
        self.knn_link = 4

    def load_vae_model(self):
        #load the general model
        self.vae_encoder = CoordVAE(
            device = self.cfg["task"]["device"],
            input_dim = self.cfg["task"]["vision"]["VAE"]["input_dims"],
            latent_dim = self.cfg["task"]["vision"]["VAE"]["latent_dims"],
        )
        # load the weight to the VAE
        self.vae_encoder.load_state_dict(th.load(self.cfg["task"]["vision"]["VAE"]["model_path"]))
        self.vae_encoder.to(self.cfg["task"]["device"])
        self.vae_encoder.eval()

    def generate_target_points(self, target_centre, init_radius):
        # we generate three concentric circle to represent the target, 3 dimensions points
        pt_list = th.zeros((85,3), device = self.device)
        pt_list[0] = target_centre
        # 1st circle
        for i in range(12):
            if i == 0:
                pt_list[i+1][0] = target_centre[0][0] + init_radius * np.cos(0)
                pt_list[i+1][1] = target_centre[0][1] + init_radius * np.sin(0)
            else:
                pt_list[i+1][0] = target_centre[0][0] + init_radius * np.cos((i * 2*np.pi) / 12)
                pt_list[i+1][1] = target_centre[0][1] + init_radius * np.sin((i * 2*np.pi) / 12)
        for i in range(24):
            if i == 0:
                pt_list[i+13][0] = target_centre[0][0] + (1.5 * init_radius * np.cos(0))
                pt_list[i+13][1] = target_centre[0][1] + (1.5 * init_radius * np.sin(0))
            else:
                pt_list[i+13][0] = target_centre[0][0] + (1.5 * init_radius * np.cos((i * 2*np.pi) / 24))
                pt_list[i+13][1] = target_centre[0][1] + (1.5 * init_radius * np.sin((i * 2*np.pi) / 24))
        for i in range(48):
            if i == 0:
                pt_list[i+37][0] = target_centre[0][0] + (2 * init_radius * np.cos(0))
                pt_list[i+37][1] = target_centre[0][1] + (2 * init_radius * np.sin(0))
            else:
                pt_list[i+37][0] = target_centre[0][0] + (2 * init_radius * np.cos((i * 2*np.pi) / 48))
                pt_list[i+37][1] = target_centre[0][1] + (2 * init_radius * np.sin((i * 2*np.pi) / 48))
        pt_list[:,[2]] = 1
        #print("pt list : " + str(pt_list))
        #print("pt list shape : " + str(pt_list.shape))

        return pt_list

    def generate_reference_pc(self, target_pcd):
        from isaaclab.utils.math import matrix_from_euler
        from isaaclab.utils.math import euler_xyz_from_quat,make_pose,matrix_from_quat,pose_inv,unmake_pose, unproject_depth, transform_points
        from isaaclab.managers import SceneEntityCfg
        from torch_geometric.data import Data
        import torch_geometric.transforms as T
        from torch_geometric.transforms import SamplePoints, KNNGraph, GridSampling
        #from isaaclab.utils.math import transform_points, unpself.reference_pcroject_depth
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
        from torch.nn.functional import normalize
        import torch.utils.dlpack
        import isaacsim.core.utils.bounds as bounds_utils
        x_min = 0.0
        y_min = -0.6
        x_max = 1.2
        y_max = 0.6

        surface_grid_size = 16

        # 0. prepare target pc
        
        new_target_pcd = target_pcd
        new_target_pcd[:,2] = target_pcd[:,2] * 0.01

        # 1. generate a surface grid
        surface_x = th.linspace(x_min, x_max, surface_grid_size, device = self.device)
        surface_y = th.linspace(y_min, y_max, surface_grid_size, device = self.device)
        surface_x, surface_y = th.meshgrid(surface_x, surface_y)

        surface_z = th.zeros(surface_grid_size * surface_grid_size, device = self.device)
        surface_pc = th.stack([surface_x.flatten(), surface_y.flatten(), surface_z.flatten()], dim = 1)

        # 2. assemble reference graph
        pc_reference = th.concatenate([surface_pc, target_pcd])
        #pcd_reference = o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(th.utils.dlpack.to_dlpack(pc_reference)))
        #o3d.io.write_point_cloud("reference_pc.ply", pcd_reference.to_legacy())

        return pc_reference

    

    def get_latent_space(self, map):
        latent_space_tensor  = self.vae_encoder.get_latent_space_vector(map)
        latent_space_tensor = th.nan_to_num(latent_space_tensor)
        return latent_space_tensor
    
    def process_point_cloud(self, cameras, _dt):
        from isaaclab.utils.math import matrix_from_euler
        from isaaclab.utils.math import euler_xyz_from_quat,make_pose,matrix_from_quat,pose_inv,unmake_pose, unproject_depth, transform_points
        from isaaclab.managers import SceneEntityCfg
        from torch_geometric.data import Data
        import torch_geometric.transforms as T
        from torch_geometric.transforms import SamplePoints, KNNGraph, GridSampling
        #from isaaclab.utils.math import transform_points, unproject_depth
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
        from torch.nn.functional import normalize
        import torch.utils.dlpack
        import isaacsim.core.utils.bounds as bounds_utils

        ee_point_base = []
        # point cloud generation from the ee point of view
        for i in range(len(cameras)):
            cameras[i].update(dt=_dt)
            images = cameras[i].data.output["distance_to_image_plane"].to(self.device)
            images = images.squeeze(dim = 0)
            points = unproject_depth(images, th.tensor(self.cfg["task"]["camera"]["intrinsic_matrix"], device = self.device), is_ortho = True)
            points = transform_points(points, th.tensor(self.cfg["task"]["camera"]["position"]["camera_" + str(i)], device = self.device), cameras[i].data.quat_w_ros[0].to(self.device))
            ee_point_base.append(points)

        pc_data = th.concatenate(ee_point_base).to(self.device)
        pcd = o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(pc_data)))
        generalBbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound=(0.1, -0.5, 0.0), max_bound=(1.1, 0.5, 0.1))
        pcd = pcd.crop(generalBbox.cuda(self.cfg["task"]["device_n"]), invert = False)
        

        bb_cache = bounds_utils.create_bbox_cache()
        x_offset = 0.015
        y_offset = 0.015
        toolBbound = bounds_utils.compute_combined_aabb(bb_cache, prim_paths=[self.cfg["task"]["tool"]["prim_path_bristles"]])
        tool_min_bound = o3d.core.Tensor(np.array([toolBbound[0] - x_offset, toolBbound[1] - y_offset, toolBbound[2]-0.01]))
        tool_max_bound = o3d.core.Tensor(np.array([toolBbound[3] + x_offset, toolBbound[4] + y_offset, toolBbound[5]+0.3]))
        toolBbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound=tool_min_bound.cuda(self.cfg["task"]["device_n"]).clone(), max_bound=tool_max_bound.cuda(self.cfg["task"]["device_n"]).clone())
        baseBbound = bounds_utils.compute_combined_aabb(bb_cache, prim_paths=[self.cfg["task"]["tool"]["prim_path_base"]])
        base_min_bound = o3d.core.Tensor(np.array([baseBbound[0]- x_offset, baseBbound[1]-y_offset, baseBbound[2]]))
        base_max_bound = o3d.core.Tensor(np.array([baseBbound[3]+ x_offset, baseBbound[4]+y_offset, baseBbound[5]]))
        baseBbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound=base_min_bound.cuda(self.cfg["task"]["device_n"]).clone(), max_bound=base_max_bound.cuda(self.cfg["task"]["device_n"]).clone())
        pcd = pcd.crop(toolBbox.cuda(self.cfg["task"]["device_n"]), invert = True)
        pcd = pcd.crop(baseBbox.cuda(self.cfg["task"]["device_n"]), invert = True)
        
        surfaceBbox = o3d.t.geometry.AxisAlignedBoundingBox(min_bound=(0.1, -0.4, 0.005), max_bound=(1.0, 0.4, 0.09))
        medium_pcd = pcd.crop(surfaceBbox.cuda(self.cfg["task"]["device_n"]), invert = False)
        #medium_pcd = medium_pcd.voxel_down_sample(voxel_size=0.05)
        #o3d.io.write_point_cloud("med_scene.ply", medium_pcd.to_legacy())
        medium = torch.utils.dlpack.from_dlpack(medium_pcd.point.positions.to_dlpack()).to(self.device)
   
        if th.numel(medium) == 0:
            medium = self.prev_medium
        else: 
            self.prev_medium = medium

        return medium

    def generate_actual_pc(self, medium_pcd, reference_pcd):
        from isaaclab.utils.math import matrix_from_euler
        from isaaclab.utils.math import euler_xyz_from_quat,make_pose,matrix_from_quat,pose_inv,unmake_pose, unproject_depth, transform_points
        from isaaclab.managers import SceneEntityCfg
        from torch_geometric.data import Data
        import torch_geometric.transforms as T
        from torch_geometric.transforms import SamplePoints, KNNGraph, GridSampling
        #from isaaclab.utils.math import transform_points, unproject_depth
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
        from torch.nn.functional import normalize
        import torch.utils.dlpack
        import isaacsim.core.utils.bounds as bounds_utils
        x_min = 0.0
        y_min = -0.6
        x_max = 1.2
        y_max = 0.6

        # 4. downsample the medium graph 
        medium = o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(th.utils.dlpack.to_dlpack(medium_pcd)))
        medium = medium.farthest_point_down_sample(num_samples = 600)
        medium_pcd = torch.utils.dlpack.from_dlpack(medium.point.positions.to_dlpack()).to(self.device)
        # 3. assemble full graph
        actual_pcd = th.concatenate([reference_pcd, medium_pcd])

        pcd_full = o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(th.utils.dlpack.to_dlpack(actual_pcd)))
        #pcd_full = pcd_full.farthest_point_down_sample(num_samples = 1800)
        #o3d.io.write_point_cloud("actual_pc.ply", pcd_full.to_legacy())

        # 4. calculate chamfer distance
        #chamfer_distance = self.get_chamfer_distance()
        return actual_pcd


        
  
    def get_chamfer_distance(self, pcd_reference, pcd_actual):
        from open3d.t.geometry import TriangleMesh, PointCloud, Metric, MetricParameters
        metric_params = MetricParameters()
        #         fscore_radius=o3d.utility.FloatVector((0.01, 0.11, 0.15, 0.18)))
        chamfer_distance = pcd_reference.compute_metrics(pcd2 = pcd_actual, 
                                                            metrics = Metric.chamfer_distance, 
                                                            params = metric_params)
        return chamfer_distance


    
    def update_heightmaps(self, medium_tensor_pc, target_pc, ee_position, ee_orientation):

        from isaaclab.utils.math import matrix_from_euler
        from isaaclab.utils.math import euler_xyz_from_quat,make_pose,matrix_from_quat,pose_inv,unmake_pose, unproject_depth, transform_points
        from isaaclab.managers import SceneEntityCfg
        from torch_geometric.data import Data
        import torch_geometric.transforms as T
        from torch_geometric.transforms import SamplePoints, KNNGraph, GridSampling
        #from isaaclab.utils.math import transform_points, unproject_depth
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
        from torch.nn.functional import normalize
        import torch.utils.dlpack
        import isaacsim.core.utils.bounds as bounds_utils
        # hmap : medium heightmap
        # tmap: target map
        # fmap : normalised full map (hmap + tmap)
        part_x = medium_tensor_pc[:,[0]] 
        part_y = medium_tensor_pc[:,[1]]
        part_x = th.mean(part_x.flatten())
        part_y = th.mean(part_y.flatten())
        particle_mean_pos = th.tensor([[part_x.item(), part_y.item()]], device = self.device)

        part_ee_pos = th.zeros((medium_tensor_pc.shape[0], 3), device = self.device)
        target_ee_pos = th.zeros((target_pc.shape[0], 3), device = self.device)
        part_ee_pos[:,[0]] = medium_tensor_pc[:,[0]].to(self.device) - ee_position[:,[0]].to(self.device)
        part_ee_pos[:,[1]] = medium_tensor_pc[:,[1]].to(self.device) - ee_position[:,[1]].to(self.device)
        part_ee_pos[:,[2]] = medium_tensor_pc[:,[2]].to(self.device)
        print("max point cloud height " + str(th.max(part_ee_pos[:,[2]])))
        target_ee_pos[:,[0]] = target_pc[:,[0]] - ee_position[:,[0]].to(self.device)
        target_ee_pos[:,[1]] = target_pc[:,[1]] - ee_position[:,[1]].to(self.device)
        target_ee_pos[:,[2]] = target_pc[:,[2]] 

        rotation_matrix = th.tensor([[th.cos(ee_orientation[2]), -th.sin(ee_orientation[2])],[th.sin(ee_orientation[2]), th.cos(ee_orientation[2])]], device = self.device)
        p = part_ee_pos[:,0:2]
        #pp = th.matmul(p[:], rotation_matrix)
        part_ee_pos[:,0:2] = p

        t = target_ee_pos[:,0:2]
        #tt = th.matmul(t[:], rotation_matrix)
        target_ee_pos[:,0:2] = t
        
        img = self.pointcloud_to_heightmap_warp(part_ee_pos,(self.cfg["task"]["camera"]["img_w"],self.cfg["task"]["camera"]["img_h"]), device = self.device)
        hmap = th.reshape(img, (self.cfg["task"]["camera"]["img_w"],self.cfg["task"]["camera"]["img_h"]))
        imax = th.max(hmap)
        imin = th.min(hmap)
        hmap = hmap - imin
        hmap = hmap / imax
        hmap = hmap * 255

        # nan guard
        default = th.isnan(hmap)
        if default.any() == True: 
            print("hmap is nan")
            hmap = self.previous_hmap
        else: 
            self.previous_hmap = hmap

        tmap = self.pointcloud_to_heightmap_warp(target_ee_pos, (self.cfg["task"]["camera"]["img_w"],self.cfg["task"]["camera"]["img_h"]), device = self.device)
        tmap = th.reshape(tmap,  (self.cfg["task"]["camera"]["img_w"],self.cfg["task"]["camera"]["img_h"]))
        tmap = tmap * 0.4 * 255

        fmap = hmap + tmap
        fmax = th.max(hmap)
        fmin = th.min(hmap)
        fmap = fmap - fmin
        fmap = fmap / fmax
        fmap = fmap * 255

        imgplot = plt.imshow(fmap.to("cpu").numpy())
        plt.savefig("fmap.png")

        fmap = fmap.reshape(1,1,self.cfg["task"]["camera"]["img_w"],self.cfg["task"]["camera"]["img_h"])

        #x_hat, _ , _ = self.vae_encoder(fmap)
        #x_hat = x_hat.squeeze(dim = 0).squeeze(dim = 0)
        #x_hat = x_hat
        #x_hat = x_hat.numpy(force=True)
        #x_hat = x_hat * 255

        #imgplot = plt.imshow(x_hat)
        #plt.savefig("fmap_output.png")
        

        return fmap, hmap, tmap

    def update_point_cloud(self, medium_tensor_pc, target_pc, ee_position, ee_orientation, Dt = 0):
        from isaaclab.utils.math import matrix_from_euler
        from isaaclab.utils.math import euler_xyz_from_quat,make_pose,matrix_from_quat,pose_inv,unmake_pose, unproject_depth, transform_points
        from isaaclab.managers import SceneEntityCfg
        from torch_geometric.data import Data
        import torch_geometric.transforms as T
        from torch_geometric.transforms import SamplePoints, KNNGraph, GridSampling, Cartesian, RemoveSelfLoops, RemoveDuplicatedEdges, VirtualNode
        from torch_geometric.utils import grid
        #from isaaclab.utils.math import transform_points, unproject_depth
        from isaaclab.sensors.camera.utils import create_pointcloud_from_depth
        from torch.nn.functional import normalize
        import torch.utils.dlpack
        import isaacsim.core.utils.bounds as bounds_utils
        # hmap : medium heightmap
        # tmap: target map
        # fmap : normalised full map (hmap + tmap)
        part_x = medium_tensor_pc[:,[0]] 
        part_y = medium_tensor_pc[:,[1]]
        part_x = th.mean(part_x.flatten())
        part_y = th.mean(part_y.flatten())
        particle_mean_pos = th.tensor([[part_x.item(), part_y.item()]], device = self.device)

        part_ee_pos = th.zeros((medium_tensor_pc.shape[0], 3), device = self.device)
        target_ee_pos = th.zeros((target_pc.shape[0], 3), device = self.device)
        part_ee_pos[:,[0]] = medium_tensor_pc[:,[0]].to(self.device) #- ee_position[:,[0]].to(self.device)
        part_ee_pos[:,[1]] = medium_tensor_pc[:,[1]].to(self.device) #- ee_position[:,[1]].to(self.device)
        part_ee_pos[:,[2]] = medium_tensor_pc[:,[2]].to(self.device)

        target_ee_pos[:,[0]] = target_pc[:,[0]] #- ee_position[:,[0]].to(self.device)
        target_ee_pos[:,[1]] = target_pc[:,[1]] #- ee_position[:,[1]].to(self.device)
        target_ee_pos[:,[2]] = target_pc[:,[2]] 
        

        new_target_ee_pos = target_ee_pos
        new_target_ee_pos[:, [2]] = target_ee_pos[:,[2]] * 0.01

        full_scene_pc = th.concatenate([part_ee_pos, new_target_ee_pos])

        structured_pc = self.get_structured_pointcloud(full_scene_pc,(self.cfg["task"]["camera"]["img_w"],self.cfg["task"]["camera"]["img_h"]), device = self.device)

        _pos = structured_pc[:,0:2]
        structured_graph = Data(x = structured_pc, pos = _pos)
        #edge_index, _ = grid(height = 64, width = 64, device = self.device)
        transform = T.Compose([KNNGraph(k = self.knn_link)])
        structured_graph = transform(structured_graph)

        #print("edge index: " + str(edge_index.shape))


        structured_graph.pos = structured_pc
        # we add the virtual node which correspond to the robot EE position
        transform = T.Compose([VirtualNode()])
        structured_graph = transform(structured_graph)
        structured_graph.x[4096] = ee_position[[0]].to(self.device)


        structured_graph = Data(x = structured_graph.x, edge_index = structured_graph.edge_index)

        # provide the relative cartesian difference of position between each nodes
        #transform = T.Compose([Cartesian(norm = False)])
        #structured_graph = transform(structured_graph)

        #structured_pcd = o3d.t.geometry.PointCloud(o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(structured_pc)))
        #o3d.io.write_point_cloud("structured_pcd.ply", structured_pcd.to_legacy())


        return structured_graph


    
    def calculate_distances(self, medium_tensor_pc, target_position, ee_position, medium_visualiser, farthest_visualiser):
        part_x = medium_tensor_pc[:,[0]] 
        part_y = medium_tensor_pc[:,[1]]
        part_x = th.mean(part_x.flatten())
        part_y = th.mean(part_y.flatten())
        particle_mean_pos = th.tensor([part_x.item(), part_y.item()], device = self.device)
        particle_target_pos = target_position[:,0:2] - particle_mean_pos
        ee_particle_pos = particle_mean_pos - ee_position
        medium_visualiser.visualize(th.tensor([[part_x, part_y, 0.05]]))

        particle_target_dist = th.norm((particle_target_pos), dim = 1)
        ee_particle_dist = th.norm((ee_particle_pos), dim = 1)

        ## farthest particle calculation
        particles_target_dists = medium_tensor_pc - target_position.to(self.device)
        particles_target_dists = particles_target_dists[:,0:2]
        particles_target_dists = th.norm(particles_target_dists, dim=1)
        particles_target_f = th.max(particles_target_dists)
        particles_target_fidx = th.argmax(particles_target_dists)
        f_part_pos = medium_tensor_pc[particles_target_fidx]
        ee_fpart_pos = f_part_pos[0:2] - ee_position
        ee_fpart_dist = th.norm((ee_fpart_pos), dim = 1)
        farthest_visualiser.visualize(th.tensor([[f_part_pos[0], f_part_pos[1], 0.05]]))

        output = {
            "ee_particle_pos": ee_particle_pos,
            "ee_particle_dist": ee_particle_dist, 
            "particle_target_pos": particle_target_pos,
            "particle_target_dist": particle_target_dist,
            "particle_mean_pos": particle_mean_pos, 
            "fpart_pos": f_part_pos,
            "ee_fpart_pos": ee_fpart_pos,
            "ee_fpart_dist": ee_fpart_dist,
        }
        return output
        #print("mediim pos : " + str(particle_mean_pos))
        #print(" ee dist : " + str(ee_position))
        #print("ee particle dist:  " + str(ee_particle_dist))

        #particles_target_dists = medium_tensor_pc - target_position.to(self.device)
        #particles_target_dists = particles_target_dists[:,0:2]heightmap.shape
        #particles_target_dists = th.norm(particles_target_dists, dim=1)
        #if th.numel(particles_target_dists) != 0:
        #    particles_target_f = th.max(particles_target_dists)
        #    particles_target_fidx = th.argmax(particles_target_dists)
        #    #print("particles target dist : " + str(particles_target_dists))
        #    particles_target_dist_mean = th.mean(particles_target_dists.flatten()) # scalar
        #    f_part_pos = medium_tensor_pc[particles_target_fidx]
        #    #self.particule_vis.visualize(th.tensor([[f_part_pos[0], f_part_pos[1], 0.01]]))
        #    ee_fpart_pos = f_part_pos[0:2] - ee_position
        #    particles_ee_dist = th.norm((ee_part_pos), dim=1)
        #    self.particles_target_f_prev = particles_target_f 
        #    self.particles_target_dist_mean_prev = particles_target_dist_mean
        #    self.particles_ee_dist_prev = particles_ee_dist
        #    self.ee_fpart_pos_prev = ee_fpart_pos
        #    self.part_target_pos_prev = part_target_pos
        #    self.ee_part_pos_prev = ee_part_pos
        #else: 
        #    particles_target_f = self.particles_target_f_prev
        #    particles_target_dist_mean = self.particles_target_dist_mean_prev
        #    particles_ee_dist = self.particles_ee_dist_prev
        #    ee_fpart_pos = self.ee_fpart_pos_prev 
       #     part_target_pos = self.part_target_pos_prev 
        #    ee_part_pos = self.ee_part_pos_prev
        
        #return ee_particle_pos, ee_particle_dist, particle_target_pos, particle_target_dist, particle_mean_pos
    
    def pointcloud_to_heightmap_warp(self,pointcloud, heightmap_size, height_range=(0, 1), device='cuda'):
        """
        Converts a 3D point cloud into a heightmap using grid_sample on the GPU.
        
        :param pointcloud: A tensor of shape (N, 3), where each row is a point (x, y, z).
        :param heightmap_size: The resolution (width, height) of the heightmap (W, H).
        :param height_range: Tuple of (min_height, max_height) to normalize z-values.
        :param device: Device to run the computation on, can be 'cuda' or 'cpu'.
        
        :return: The generated heightmap of shape (1, 1, H, W) after grid_sample.
        """
        # Move pointcloud to the specified device
        pointcloud = pointcloud.to(device)


        # Normalize x and y coordinates to fit within heightmap
        x_min = -0.60
        y_min = -0.60
        x_max = 0.60
        y_max = 0.60
        #x_min, y_min = pointcloud[:, :2].min(dim=0).values
        #x_max, y_max = pointcloud[:, :2].max(dim=0).values
        
        # Scale the x, y coordinates to fit within the heightmap grid size
        x_norm = (pointcloud[:, 0] - x_min) / (x_max - x_min) * (heightmap_size[0] - 1)
        y_norm = (pointcloud[:, 1] - y_min) / (y_max - y_min) * (heightmap_size[1] - 1)
        
        # Normalize z values (height) to be within the specified height range
        z_min, z_max = height_range
        z_norm = (pointcloud[:, 2] - z_min) / (z_max - z_min)

        ptc = th.zeros((pointcloud.shape[0], 3), device = self.device)
        ptc[:,0] = x_norm.reshape(x_norm.shape[0])
        ptc[:,1] = y_norm.reshape(y_norm.shape[0])
        ptc[:,2] = z_norm.reshape(z_norm.shape[0])
        
        # Initialize heightmap tensor (with zeros initially)
        #heightmap = th.zeros((1, 1, *heightmap_size), dtype=th.float32, device=device)
        # Allocate memory for the heightmap
        heightmap = wp.zeros((heightmap_size[0],heightmap_size[1]), dtype=float, device=self.device)
        # Create a Warp kernel to generate the heightmap
        @wp.kernel
        def generate_heightmap(points: wp.array(dtype=wp.vec3), heightmap: wp.array2d(dtype=float)):
            tid = wp.tid()
            x = int(points[tid][0])
            y = int(points[tid][1])
            z = points[tid][2]

            wp.atomic_max(heightmap, x,y, z)
        
        # Launch the kernel
        wp.launch(generate_heightmap, dim=ptc.shape[0], inputs=[ptc, heightmap],  device=self.device)
        
        # Generate a grid for grid_sample
        grid_y, grid_x = th.meshgrid(th.arange(heightmap_size[1], device=device), 
                                        th.arange(heightmap_size[0], device=device))
        
        grid = th.stack((grid_x, grid_y), dim=-1)  # Shape: (H, W, 2)
        
        # Normalize the grid to [-1, 1] for grid_sample
        grid = (grid / th.tensor(heightmap_size, device=device).float() - 0.5) * 2
        
        # Add batch and channel dimensions to match grid_sample's requirements
        grid = grid.unsqueeze(0)

        
        # convert to Torch tensor
        heightmap = wp.to_torch(heightmap)
        heightmap = th.reshape(heightmap, (1,1,heightmap_size[0],heightmap_size[1]))
        
        # Use grid_sample to sample the heightmap
        sampled_heightmap = th.nn.functional.grid_sample(heightmap, grid)
        
        return sampled_heightmap

    def get_structured_pointcloud(self, pointcloud, map_size,upscale = 100, device = "cuda"):
                # Move pointcloud to the specified device
        pointcloud = pointcloud.to(device)
        # Normalize x and y coordinates to fit within heightmap
        x_min = 0.1
        y_min = -0.50
        x_max = 1.1
        y_max = 0.50
        #x_min, y_min = pointcloud[:, :2].min(dim=0).values
        #x_max, y_max = pointcloud[:, :2].max(dim=0).values
        
        # Scale the x, y coordinates to fit within the heightmap grid size
        x_norm = (pointcloud[:, 0] - x_min) / (x_max - x_min) * (map_size[0] - 1)
        y_norm = (pointcloud[:, 1] - y_min) / (y_max - y_min) * (map_size[1] - 1)
        
        # Normalize z values (height) to be within the specified height range
        z_norm = pointcloud[:, 2]

        ptc = th.zeros((pointcloud.shape[0], 3), device = self.device)
        ptc[:,0] = x_norm.reshape(x_norm.shape[0])
        ptc[:,1] = y_norm.reshape(y_norm.shape[0])
        ptc[:,2] = z_norm.reshape(z_norm.shape[0])
        
        # Initialize heightmap tensor (with zeros initially)
        #heightmap = th.zeros((1, 1, *heightmap_size), dtype=th.float32, device=device)
        # Allocate memory for the heightmap
        heightmap = wp.zeros((map_size[0],map_size[1]), dtype=float, device=self.device)
        # Create a Warp kernel to generate the heightmap
        @wp.kernel
        def generate_heightmap(points: wp.array(dtype=wp.vec3), heightmap: wp.array2d(dtype=float)):
            tid = wp.tid()
            x = int(points[tid][0])
            y = int(points[tid][1])
            z = points[tid][2]

            wp.atomic_max(heightmap, x,y, z)
        
        # Launch the kernel
        wp.launch(generate_heightmap, dim=ptc.shape[0], inputs=[ptc, heightmap],  device=self.device)

        point_cloud = th.zeros((map_size[0] * map_size[1], 3), device = self.device)

        heightmap = wp.to_torch(heightmap)

        #x_coords = th.arange(map_size[1]).unsqueeze(1).repeat(1, map_size[1])
        #y_coords = th.arange(map_size[0]).unsqueeze(1).repeat(1, map_size[0])

        x_coords, y_coords = th.meshgrid(th.arange(map_size[1], device=self.device), 
                                        th.arange(map_size[0], device=self.device))
    
        x_coords = x_coords.to(self.device)
        y_coords = y_coords.to(self.device)
        heightmap = heightmap.to(self.device)

        x_coords = x_coords.flatten()
        y_coords = y_coords.flatten()
        heightmap = heightmap.flatten()

        xx = ((x_coords.view(-1, 1).type(th.float32) - (map_size[0] / 2)) / map_size[0]) * (x_max - x_min)
        yy = ((y_coords.view(-1, 1).type(th.float32) - (map_size[1] / 2)) / map_size[1]) * (y_max - y_min)
        zz = heightmap.view(-1, 1).type(th.float32) 

        grid = th.stack((xx, yy, zz), dim=-1)  # Shape: (H, W, 2)
        grid = grid.view(-1, 3)

        return grid



    


    
    



    
