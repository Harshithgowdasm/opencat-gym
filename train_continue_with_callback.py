from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from opencat_gym_env import OpenCatGymEnv
from opencat_step_gym_env import OpenCatStepGymEnv
from callback_save_best_model import SaveBestModelCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Create OpenCatGym environment from class and check if structure is correct
# env = OpenCatGymEnv()
# check_env(env)

model_name = "PPO_2"
model_path = f"trained/{model_name}/final_model"

TASK = "step"  # ["gait", "step"]

if __name__ == "__main__":
    # Set up number of parallel environments
    parallel_env = 8
    if TASK == "gait":
        env = make_vec_env(
            OpenCatGymEnv, n_envs=parallel_env, vec_env_cls=SubprocVecEnv
        )

    elif TASK == "step":
        env = make_vec_env(
            OpenCatStepGymEnv, n_envs=parallel_env, vec_env_cls=SubprocVecEnv
        )

    # Define PPO agent with custom network architecture
    custom_arch = dict(net_arch=[256, 256])

    # Path to save the best model
    save_path = f"trained/{model_name}"

    # Create the callback to save the best model
    save_best_callback = SaveBestModelCallback(
        check_freq=parallel_env, save_path=save_path
    )

    # Load model to continue previous training
    model = PPO.load(
        model_path,
        env,
        policy_kwargs=custom_arch,
        n_steps=int(2048 * 8 / parallel_env),
        verbose=1,
        tensorboard_log="trained/tensorboard_logs/",
    ).learn(2e6)
    model.save("trained/final_model_2")
