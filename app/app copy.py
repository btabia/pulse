
import asyncio
import numpy as np


class App():
    def __init__(self, cfg = None) -> None:
        self.cfg = cfg
        import argparse
        from omni.isaac.lab.app import AppLauncher
        print("app dict" + str(self.cfg["app"]))
        app_launcher = AppLauncher(vars(self.cfg["app"]))
        self._simulation_app = AppLauncher(self.config)
    
    def get_simulation_context(self):
        from omni.isaac.lab.sim import SimulationCfg, SimulationContext

        self._physics_dt = 1 / self.app_config["app"]["simulation_context"]["physics_freq"]
        self.rendering_dt = 1 / self.app_config["app"]["simulation_context"]["rendering_freq"]

        sim_cfg = SimulationCfg(
            physics_prim_path = self.cfg["app"]["simulation_context"]["physics_prim_path"],
            physics_dt = self._physics_dt,
            rendering_dt = self.rendering_dt,
            set_defaults = self.cfg["app"]["simulation_context"]["set_defaults"],
            stage_units_in_meters = self.cfg["app"]["simulation_context"]["stage_unit_in_meters"],
            device = self.cfg["app"]["simulation_context"]["device"],
            backend = self.cfg["app"]["simulation_context"]["backend"],
        )
        self.sim = SimulationContext(sim_cfg)

        return self.sim
    
    def run_simulator(self):
        self.sim.reset()
        print("[INFO]: Setup complete...")

        while self._simulation_app.is_running():
            self.sim.step()



