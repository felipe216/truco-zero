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
        play = 0
        print("starting...")
        while not done:
            print(f"Actual state: {status['your cards']}, {status['bot_cards']}")

            if play >= 2:
                second_action = True
            else:
                second_action = False

            if turn == 0:
                play+=1
                action_agent = model.predict(obs, deterministic=True)[0]
                print("Action agent:", action_agent)

                obs, reward, done, truncated, info = env.step(player_turn=turn+1,action=action_agent, second_action=second_action)
                turn = 1
            else:
                play+=1
                print("Choose your action:\n")
                action_human = int(input("0, 1, 2 to play a card; 3 to truca; 4 to pass; 5 to raise: "))

                obs, reward, done, truncated, info = env.step(player_turn=turn+1,action=action_human, second_action=second_action)
                turn = 0
            
            status = env._get_human_observation()
            if done:
                break


    print("Fim do jogo!")

