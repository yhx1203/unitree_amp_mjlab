from mjlab.tasks.registry import register_mjlab_task

from src.tasks.amp.rl import AmpOnPolicyRunner

from .env_cfgs import unitree_g1_amp_flat_env_cfg
from .rl_cfg import unitree_g1_amp_ppo_runner_cfg

register_mjlab_task(
  task_id="Unitree-G1-AMP-Flat",
  env_cfg=unitree_g1_amp_flat_env_cfg(),
  play_env_cfg=unitree_g1_amp_flat_env_cfg(play=True),
  rl_cfg=unitree_g1_amp_ppo_runner_cfg(),
  runner_cls=AmpOnPolicyRunner,
)

