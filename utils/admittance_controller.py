import numpy as np 
import torch as th 


class AdmittanceController():
    def __init__(self, cfg): 
        self.cfg = cfg
        self.device = self.cfg["task"]["device"]
        self.force_command = th.tensor([0,0,0], device = self.device)
        self.Dpos_prev = th.tensor([[0,0,0]], device = self.device)
        self.Dvel_prev = th.tensor([[0,0,0]], device = self.device)
        self.M = th.tensor(self.cfg["task"]["controller"]["M"], device = self.device)
        self.K = th.tensor(self.cfg["task"]["controller"]["K"], device = self.device)
        self.B = th.tensor(self.cfg["task"]["controller"]["B"], device = self.device)
        self.I = th.ones((1,3), device = self.device)

    def set_force_command(self, command):
        self.force_command = command
    
    def set_force_observation(self, obs):
        self.force_observation = obs

    def set_delta_pos(self, delta_pos):
        self.Dpos = delta_pos
    
    def set_delta_vel(self, delta_vel): 
        self.Dvel = delta_vel
    
    def compute(self, dt):
        f_delta = self.force_command - self.force_observation
        dt_ = th.tensor([[dt, dt, dt]], device = self.device)
        dt2_ = th.tensor([[dt*dt, dt*dt, dt*dt]], device = self.device)
        pa = (dt2_/self.M) * f_delta
        pb = (dt_ - (self.B/self.M)*dt2_)*self.Dvel
        pc = (self.I - (self.K/self.M)*dt2_) * self.Dpos 
        output = pa + pb + pc
        return output


