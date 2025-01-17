import time
import pybullet as p
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv

# Create OpenCatGym environment from class
parallel_env = 1
env = make_vec_env(OpenCatGymEnv)

RL_ALGORITHM = "TD3"  # ["PPO", "DDPG", "TD3"]
model_name = "best_model_step_25100_reward_92.476361"

if RL_ALGORITHM == "PPO":
    model = PPO.load(f"trained/{model_name}")

if RL_ALGORITHM == "DDPG":
    model = DDPG.load(f"trained/{model_name}")

if RL_ALGORITHM == "TD3":
    model = TD3.load(f"trained/{model_name}")

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
