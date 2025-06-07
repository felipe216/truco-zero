import random
from truco.env import TrucoEnv
from stable_baselines3 import PPO


def play_game():
    model = PPO.load("ppo_truco")
    turn = random.randint(0, 1)
    status = {'agent1_matches_won': 0, 'agent2_matches_won': 0, 'agent1_score': 0, 'agent2_score': 0, 'rodada': 0, 'truco_called': False, 'truco_value': 0}
    env = TrucoEnv(mode='game')
    while not status['agent1_matches_won'] >= 12 or not status['agent2_matches_won'] >= 12:
        
        obs, status = env.reset()
        done = False
        print("starting...")
        while not done:
            print(f"Actual state: {status}")

            if turn == 0:
                action_agent = model.predict(obs, deterministic=True)[0]
                print("Action agent:", action_agent)
                # Passa ação do agente para o ambiente
                obs, reward, done, truncated, info = env.step(action_agent)
                turn = 1
            else:
                print("Choose your action:\n")
                action_human = int(input("0, 1, 2 to play a card; 3 to truca; 4 to pass; 5 to raise: "))

                obs, reward, done, truncated, info = env.step(action_human)
                turn = 0
                status = env._get_human_observation()
            if done:
                break


    print("Fim do jogo!")

