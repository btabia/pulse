from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, interface_config_loader, LulaKinematicsSolver
from isaacsim.core.prims import SingleArticulation, Articulation
from typing import Optional
from isaacsim.robot_motion.motion_generation import KinematicsSolver
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver

import numpy as np
import carb


class RobotKinematics:
    def __init__(self, 
                 cfg, 
                 articulation: Articulation, 
                ):
        self.cfg = cfg
        self._articulation = articulation
        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path = self.cfg["task"]["kinematics"]["robot_description_path"],
            urdf_path = self.cfg["task"]["kinematics"]["urdf_path"],
        )
        end_effector_name = self.cfg["task"]["kinematics"]["ee_link"]
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(self._articulation,self._kinematics_solver, end_effector_name)


    
    def compute_ik(self, joint_positions, position_cmd, orientation_cmd):
        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(joint_positions, position_cmd, orientation_cmd)
        return action, success


