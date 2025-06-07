from agents.random_agent import RandomAgent
from truco.env import TrucoEnv

def simulate_game():
    env = TrucoEnv()
    obs, _ = env.reset()
    agente_random = RandomAgent()

    done = False
    while not done:
        actions = env.action_space.sample()
        action = agente_random.choose_action(obs, actions)
        obs, reward, done, _, _ = env.step(action)
        env.render()

    print("Reward:", reward)
