from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from opencat_gym_env import OpenCatGymEnv
from callback_save_best_model import SaveBestModelCallback

# Create OpenCatGym environment from class and check if structure is correct
# env = OpenCatGymEnv()
# check_env(env)

if __name__ == "__main__":
    # Set up number of parallel environments
    parallel_env = 8
    env = make_vec_env(OpenCatGymEnv, n_envs=parallel_env, vec_env_cls=SubprocVecEnv)

    # Define PPO agent with custom network architecture
    custom_arch = dict(net_arch=[256, 256])

    # Path to save the best model
    save_path = "trained"

    # Create the callback to save the best model
    save_best_callback = SaveBestModelCallback(check_freq=100, save_path=save_path)

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        seed=42,
        policy_kwargs=custom_arch,
        n_steps=int(2048 * 8 / parallel_env),
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard_logs",  # Log training data
    )

    # Train the model, using the callback to record rewards and save the best model
    model.learn(total_timesteps=2e6, callback=save_best_callback)

    # Save the final trained model
    model.save(f"{save_path}/final_model")

    # Load model to continue previous training
    # model = PPO.load("trained/final_model",
    #                   env, policy_kwargs=custom_policy_kwargs,
    #                   n_steps=int(2048*8/parallel_env), verbose=1,
    #                   tensorboard_log="trained/tensorboard_logs/").learn(2e6)
    # model.save("trained/final_model_2")
