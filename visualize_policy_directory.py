import time
import os
import pybullet as p
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv
from opencat_step_gym_env import OpenCatStepGymEnv

TASK = "step"  # ["gait", "step"]

# Create OpenCatGym environment from class
parallel_env = 1
if TASK == "gait":
    env = make_vec_env(
        OpenCatGymEnv, n_envs=parallel_env
    )
elif TASK == "step":
    env = make_vec_env(
        OpenCatStepGymEnv, n_envs=parallel_env
    )

RL_ALGORITHM = "PPO"
MODEL = "PPO_step_1"

policy_dir = f"trained/{MODEL}"

for model_name in os.listdir(policy_dir):

    if RL_ALGORITHM == "PPO":
        model = PPO.load(os.path.join(policy_dir, model_name))

    if RL_ALGORITHM == "DDPG":
        model = DDPG.load(os.path.join(policy_dir, model_name))

    if RL_ALGORITHM == "TD3":
        model = TD3.load(os.path.join(policy_dir, model_name))

    print(f"Model: {model_name}")

    obs = env.reset()
    sum_reward = 0

    for i in range(500):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        sum_reward += reward
        env.render(mode="human")
        if done:
            print("Reward", sum_reward[0])
            sum_reward = 0
            obs = env.reset()
