from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from truco.env import TrucoEnv



def train_agent():
    env = TrucoEnv()

    check_env(env, warn=True)

    vec_env = DummyVecEnv([lambda: TrucoEnv()])

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)

    model.save('ppo_truco')

    print('Training finished')