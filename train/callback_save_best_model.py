from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class SaveBestModelCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super(SaveBestModelCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Check if it's time to evaluate the reward
        if self.n_calls % self.check_freq == 0:
            # Extract episode rewards
            episode_rewards = []
            for info in self.locals["infos"]:
                if "episode" in info.keys():
                    episode_rewards.append(info["episode"]["r"])

            # Calculate the mean reward
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards)

                if self.verbose > 0:
                    print(f"Step: {self.n_calls} | Mean Reward: {mean_reward}")

                # Save the model if we have a new best mean reward
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    if self.verbose > 0:
                        print(
                            f"New best mean reward: {self.best_mean_reward}. Saving model at step {self.n_calls}..."
                        )

                    self.model.save(
                        f"{self.save_path}/best_model_step_{self.n_calls}_{episode_rewards[-1]}"
                    )

        return True
