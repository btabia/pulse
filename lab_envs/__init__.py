from gymnasium.envs.registration import register
from lab_envs.base_env import BaseEnv

register(
     id="BaseEnv-v0",
     entry_point=BaseEnv,
     disable_env_checker=True,
)