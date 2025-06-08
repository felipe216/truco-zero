from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from truco.env import TrucoEnv
from train.reward_callback import RewardLoggingCallback


def train_agent():
    check_env(TrucoEnv(), warn=True)


    vec_env = DummyVecEnv([lambda: TrucoEnv()])

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path='./checkpoints/', name_prefix='truco_agent')
    callback = RewardLoggingCallback()

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=10_000, callback=[callback, checkpoint_callback])

    model.save('ppo_truco')

    print('Training finished')
    callback.plot_rewards()