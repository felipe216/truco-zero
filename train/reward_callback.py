import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0, window_size=50):
        super().__init__(verbose)
        self.episode_rewards = []
        self.window_size = window_size

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is not None and infos is not None:
            for done, info in zip(dones, infos):
                if done and "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
        return True

    def plot_rewards(self):
        plt.figure(figsize=(12, 6))
        episodes = np.arange(len(self.episode_rewards))
        rewards = np.array(self.episode_rewards)

        # Média móvel
        if len(rewards) >= self.window_size:
            moving_avg = np.convolve(rewards, np.ones(self.window_size)/self.window_size, mode='valid')
            plt.plot(episodes[:len(moving_avg)], moving_avg, label=f'Média móvel ({self.window_size})', color='blue')

        # Recompensas reais
        plt.plot(episodes, rewards, alpha=0.3, label='Recompensa por episódio', color='gray')

        plt.title("Recompensa por Episódio durante o Treinamento")
        plt.xlabel("Episódio")
        plt.ylabel("Recompensa")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reward_progress.png", dpi=300)
        plt.show()