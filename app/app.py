
import asyncio
import numpy as np
class App():
    def __init__(self, cfg = None) -> None:
        self.cfg = cfg
        import argparse
        from isaaclab.app import AppLauncher
        
        
        print("app dict" + str(self.cfg))
        app_launcher = AppLauncher(self.cfg["app"])

        self.simulation_app = app_launcher.app


    def get_sim_app(self):
        return self._simulation_app

    def is_running(self):
        self._simulation_app.is_running()
    
    def close(self):
        self._simulation_app.close()