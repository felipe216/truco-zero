import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_step(self) -> bool:
        done_array = self.locals.get("dones")
        rewards_array = self.locals.get("rewards")
        if done_array is not None and rewards_array is not None:
            for done, reward in zip(done_array, rewards_array):
                if done:
                    ep_rew = self.locals["infos"][0].get("episode")["r"]
                    self.episode_rewards.append(ep_rew)
        return True

    def plot_rewards(self):
        plt.plot(self.episode_rewards)
        plt.xlabel("Episódios")
        plt.ylabel("Recompensa")
        plt.title("Recompensa por Episódio durante o Treinamento")
        plt.grid(True)
        plt.savefig("reward_progress.png")
        plt.show()
