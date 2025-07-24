import numpy as np
import random
import torch as th


class ObjectManager():
    def __init__(self, cfg, obj_prim_path, device): 
        from isaacsim.core.prims import XFormPrim, RigidPrim, GeometryPrim
        self.cfg = cfg
        self.device = device
        self.obj_prim_path = obj_prim_path
        self.surface_prim = self.set_surface_prim()

        self.training_prim_paths = self.cfg["training_set"]["prim_path"]
        self.testing_prim_paths = self.cfg["testing_set"]["prim_path"]
        self.fullset_prim_paths = self.cfg["full_set"]["prim_path"]

        self.training_init_poses = self.cfg["training_set"]["init_pos"]
        self.testing_init_poses = self.cfg["testing_set"]["init_pos"]
        self.fullset_init_poses = self.cfg["full_set"]["init_pos"]

    def randomize_properties(self):
        rng =  np.random.default_rng()
        # 1. randomly choose between the different shape type cube, capsule, cylinder
        if self.cfg["type"] == "training":
            range_idx = len(self.training_prim_paths) - 1   
        elif self.cfg["type"] == "testing": 
            range_idx = len(self.testing_prim_paths) - 1
        elif self.cfg["type"] == "fullset":
            range_idx = len(self.fullset_prim_paths) - 1
        self.obj_idx = rng.integers(low = 0, high = range_idx, endpoint=True)
        print("Object idx: " + str(self.obj_idx))
    

    def remove_object_from_work_table(self, prim): 
        from isaacsim.core.prims import XFormPrim, RigidPrim, GeometryPrim
        import isaacsim.core.utils.prims as prims_utils
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicCapsule

        if self.cfg["type"] == "training":
            prim = RigidPrim(
                prim_paths_expr = self.training_prim_paths[int(self.obj_idx)],
            )

        elif self.cfg["type"] == "testing":
            prim = RigidPrim(
                prim_paths_expr = self.testing_prim_paths[int(self.obj_idx)],
            )
        elif self.cfg["type"] == "fullset":
            prim = RigidPrim(
                prim_paths_expr = self.fullset_prim_paths[int(self.obj_idx)],
            )
       
        if self.cfg["type"] == "training":
            print("obj pose init: " + str(th.tensor([self.training_init_poses[int(self.obj_idx)]], device = self.device).to(th.float32)))
            print("object prim : " + self.training_prim_paths[int(self.obj_idx)])
            prim.set_world_poses(
            positions = th.tensor([self.training_init_poses[int(self.obj_idx)]], device = self.device).to(th.float32),
            orientations = th.tensor([[1,0,0,0]], device = self.device).to(th.float32)
                )
            print("cube set on initial position")
            
        elif self.cfg["type"] == "testing":
            print("obj pose init: " + str(th.tensor([self.fullset_init_poses[int(self.obj_idx)]], device = self.device).to(th.float32)))
            print("object prim : " + self.fullset_prim_paths[int(self.obj_idx)])
            prim.set_world_poses(
            positions = th.tensor([self.testing_init_poses[int(self.obj_idx)]], device = self.device).to(th.float32),
            orientations = th.tensor([[1,0,0,0]], device = self.device).to(th.float32)
                )
            print("cylinder set on initial position")

        elif self.cfg["type"] == "fullset":
            print("obj pose init: " + str(th.tensor([self.fullset_init_poses[int(self.obj_idx)]], device = self.device).to(th.float32)))
            print("object prim : " + self.fullset_prim_paths[int(self.obj_idx)])
            prim.set_world_poses(
            positions = th.tensor([self.fullset_init_poses[int(self.obj_idx)]], device = self.device).to(th.float32),
            orientations = th.tensor([[1,0,0,0]], device = self.device).to(th.float32)
                )
            print("object set on initial position")

    def redefine_object(self, position = th.tensor([[1.35,0.0,0.05]]), heading = 0):
        from isaacsim.core.prims import XFormPrim, RigidPrim, GeometryPrim
        import isaacsim.core.utils.prims as prims_utils
        import isaacsim.core.utils.stage as stage_utils
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicCapsule
        from isaaclab.sim.utils import make_uninstanceable
        from isaaclab.utils.math import quat_from_euler_xyz

        if self.cfg["type"] == "training":
            prim = RigidPrim(
                prim_paths_expr = self.training_prim_paths[int(self.obj_idx)],
            )
            print("Selected object (training): " + str(self.training_prim_paths[int(self.obj_idx)]))

            pos_init = th.tensor([self.training_init_poses[int(self.obj_idx)]], device = self.device).to(th.float32)
            ori_init = th.tensor([[1,0,0,0]], device = self.device).to(th.float32)
            position[:,2] = pos_init[:,2]
            print("object idx: " + str(self.obj_idx))
            new_ori = quat_from_euler_xyz(th.tensor([0], device = self.device), 
                                th.tensor([0], device = self.device),
                                th.tensor([heading], device = self.device))
            prim.set_world_poses(
                    positions = position.to(th.float32),
                    orientations = new_ori.to(th.float32),
                )
        elif self.cfg["type"] == "testing":
            prim = RigidPrim(
                prim_paths_expr = self.testing_prim_paths[int(self.obj_idx)],
            )
            print("Selected object (testing): " + str(self.training_prim_paths[int(self.obj_idx)]))
        
        elif self.cfg["type"] == "fullset":
            prim = RigidPrim(
                prim_paths_expr = self.fullset_prim_paths[int(self.obj_idx)],
            )
            print("Selected object (fullset): " + str(self.fullset_prim_paths[int(self.obj_idx)]))

            pos_init = th.tensor([self.fullset_init_poses[int(self.obj_idx)]], device = self.device).to(th.float32)
            ori_init = th.tensor([[1,0,0,0]], device = self.device).to(th.float32)
            position[:,2] = pos_init[:,2]

            new_ori = quat_from_euler_xyz(th.tensor([0], device = self.device), 
                                            th.tensor([0], device = self.device),
                                            th.tensor([heading], device = self.device))
            print("object idx: " + str(self.obj_idx))
            print("new position set: " + str(position))
            print("new position 2: " + str(pos_init[:,2]))
            print("new orientation set: " + str(ori_init))
            prim.set_world_poses(
                    positions = position.to(th.float32),
                    orientations = new_ori.to(th.float32),
                )
        
        return prim

    def set_surface_prim(self):
        from isaacsim.core.prims import  GeometryPrim
        surface_prim = GeometryPrim(
            prim_paths_expr = self.cfg["surface"]["prim_path"],
            collisions = th.tensor([[True]], device = self.device),
        )
        return surface_prim

    def randomize_surface_friction(self): 
        from isaacsim.core.api.materials import PhysicsMaterial
        df_range = np.array(self.cfg["surface"]["dynamic_friction"])
        dynamic_friction = np.random.uniform(low= df_range[0], high= df_range[1])

        sf_range = np.array(self.cfg["surface"]["static_friction"])
        static_friction = np.random.uniform(low= sf_range[0], high= sf_range[1])

        material = PhysicsMaterial(
            prim_path = "/World/physics_material/cube_material",
            dynamic_friction = dynamic_friction,
            static_friction = static_friction,
        )
        self.surface_prim.apply_physics_materials(material)
        print("Dynamic friction applied: " + str(dynamic_friction))
        print("Static friction applied: " + str(static_friction))
        



