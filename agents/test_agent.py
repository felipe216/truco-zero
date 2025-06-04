from stable_baselines3 import PPO
from truco.env import TrucoEnv


def run_agent():
    model = PPO.load("ppo_truco")
    env = TrucoEnv()
    obs, _ = env.reset()

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()

    print("Recompensa final:", reward)
