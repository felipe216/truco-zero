from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from truco.env import TrucoEnv
from train.reward_callback import RewardLoggingCallback


def train_agent():
    def make_env():
        return Monitor([make_env])


    vec_env = DummyVecEnv([lambda: TrucoEnv()])
    vec_env = VecMonitor(vec_env) 

    checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path='./checkpoints/', name_prefix='truco_agent')
    callback = RewardLoggingCallback()

    model = PPO("MlpPolicy", vec_env, verbose=1)
    model = PPO.load("models/ppo_truco", env=vec_env)
    model.learn(total_timesteps=10_000, callback=[callback, checkpoint_callback])

    model.save('models/ppo_truco')

    print('Training finished')
    callback.plot_rewards()