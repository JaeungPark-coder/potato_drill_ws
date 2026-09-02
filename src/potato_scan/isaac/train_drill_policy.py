"""Trains the drill-approach RL policy (potato_scan/rl/drill_policy_spec.py /
drill_policy_backend.py) inside Isaac Sim's own physics, via
rl_drill_train_env.IsaacDrillEnv.

NOT a ROS2 node -- this needs Isaac Sim's own Kit Python runtime, so it's
launched directly with Isaac Sim's bundled interpreter, exactly like
isaac_scene.py:

    <isaac-sim-install-dir>/python.sh src/potato_scan/isaac/train_drill_policy.py

Prerequisites (once):
  1. colcon build --symlink-install && source install/setup.bash
     -- so `from potato_scan...` (rl/drill_policy_spec.py, drill_task_planner.py)
     resolves inside rl_drill_train_env.py. Same requirement isaac_scene.py's
     own `import rclpy` already has.
  2. <isaac-sim-install-dir>/python.sh -m pip install stable-baselines3 gymnasium

Episodes here are short (one real approach + insertion attempt, or a
handful of unreachable tries), so this should train faster wall-clock
than the scan policy, but it's still a single real-time Isaac Sim
instance -- TOTAL_TIMESTEPS below is a starting point to iterate on, not
a guaranteed-converged budget; watch the tensorboard log
(`tensorboard --logdir rl_logs/drill`) and extend/shorten it from there.
Written and reasoned about WITHOUT the ability to actually run Isaac Sim
in the environment this was authored in -- treat this as a solid first
draft, not verified to run.
"""
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from rl_drill_train_env import IsaacDrillEnv

TOTAL_TIMESTEPS = 100_000
CHECKPOINT_EVERY = 2_000
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "rl_logs", "drill")


def main():
    env = IsaacDrillEnv()

    checkpoint_cb = CheckpointCallback(
        save_freq=CHECKPOINT_EVERY,
        save_path=os.path.join(MODELS_DIR, "drill_checkpoints"),
        name_prefix="drill_policy",
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR)
    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_cb)
    finally:
        model.save(os.path.join(MODELS_DIR, "drill_policy_final"))
        env.close()

    print(f"done -- final model saved to {os.path.join(MODELS_DIR, 'drill_policy_final.zip')}")
    print("point drill_controller.rl_model_path at that file and set approach_policy: \"rl\" to use it.")


if __name__ == "__main__":
    main()
