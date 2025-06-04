from truco.env import TrucoEnv
from stable_baselines3 import PPO


def play_game():
    env = TrucoEnv()
    model = PPO.load("ppo_truco")
    obs = env.reset()[0]
    done = False

    while not done:
        print(f"Estado atual: {obs}")

        # Jogador humano escolhe ação
        action_human = int(input("Escolha sua ação (0, 1, 2...): "))

        # Passa a ação do humano para o ambiente
        obs, reward, done, truncated, info = env.step(action_human)
        if done:
            break

        # Agente escolhe ação com base na observação atual
        action_agent = model.predict(obs, deterministic=True)[0]

        # Passa ação do agente para o ambiente
        obs, reward, done, truncated, info = env.step(action_agent)

    print("Fim do jogo!")
