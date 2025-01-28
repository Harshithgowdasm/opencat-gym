import time
import pybullet as p
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from opencat_gym_env import OpenCatGymEnv
from opencat_step_gym_env import OpenCatStepGymEnv

# Create OpenCatGym environment from class
parallel_env = 1
TASK = "gait"  # ["gait", "step"]

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

RL_ALGORITHM = "TD3"  # ["PPO", "DDPG", "TD3"]
model_name = "best_tranied_models/best_model_gait_DDPG"

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
