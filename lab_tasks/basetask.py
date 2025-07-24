from __future__ import annotations
import torch
from collections.abc import Sequence
import torch as th

class Basetask():
    def __init__(self):
        self.episode_idx = 0
        self.logging_data = {}
        return
    
    def set_sim(self, sim): 
        self.sim = sim

    def set_up_scene(self):
        pass

    def get_observations(self):
        pass

    def reset(self, env_ids: Sequence[int] | None):
        pass

    def is_done(self) -> tuple[torch.Tensor, torch.Tensor]:
        pass

    def scale_actions(self, actions: torch.Tensor) -> None:
        pass

    def apply_action(self, actions):
        pass

    def compute_rewards(self):
        pass

    def set_datalogger(self, datalogger):
        self.setlogger = datalogger
    
    def get_logging_data(self):
        return self.logging_data
    
    def calculate_target_position(self,current_target_pos, step, timestep, velocity, device):
        _increment_x = th.tensor([velocity[0] * timestep], device = device)
        _increment_y = th.tensor([velocity[1] * timestep], device = device)
        target_new_pos = th.zeros(3, device = device)
        if 0 <= step <= 127:
            increment_x = th.tensor([0.0], device = device)
            increment_y = _increment_y
        elif 128 <= step <= 224:
            increment_x = -_increment_x
            increment_y = th.tensor([0.0], device = device)
        elif 226 <= step <= 290: 
            increment_x = th.tensor([0.0], device = device)
            increment_y = -_increment_y
        elif 291 <= step <= 383: 
            increment_x = _increment_x
            increment_y = th.tensor([0.0], device = device)
        else: 
            increment_x = th.tensor([0.0], device = device)
            increment_y = th.tensor([0.0], device = device)
        target_new_pos[0] = current_target_pos[0][0] - increment_x[0]
        target_new_pos[1] = current_target_pos[0][1] - increment_y[0]
        target_new_pos[2] = current_target_pos[0][2] 

        target_new_pos = target_new_pos.unsqueeze(0)
        
        return target_new_pos


    



