import gymnasium as gym
import numpy as np
from truco.env import TrucoEnv
# from stable_baselines3 import PPO # Descomente se quiser carregar um modelo treinado

def play_game_vs():
    env = TrucoEnv(mode="two_agents")


    print("Iniciando jogo de Truco: Humano (Agente 2) vs. Humano (Agente 1)")
    print("------------------------------------------------------------")

    while True: # Loop para múltiplos jogos/episódios
        obs, info = env.reset()
        done = False
        
        print("\n--- Novo Jogo Iniciado ---")
        print("--------------------------")

        while not done: # Loop para um único jogo/episódio
            current_player = env.current_player
            player_name = f"Agente {current_player}"

            # Obtém observação e informações para a perspectiva do jogador atual
            obs = env._get_observation(current_player)
            info = env._get_info(current_player)

            print(f"\n--- Turno do {player_name} ---")
            print(f"Rodada (Mão atual): {info['round']}")
            print(f"Valor do Truco: {info['truco value']}")
            print(f"Truco Chamado: {'Sim' if info['trucado'] else 'Não'}")
            print(f"Placar da Partida: Agente 1: {env.game.current_match.agent1_score} pontos, Agente 2: {env.game.current_match.agent2_score} pontos")
            print(f"Partidas Ganhas no Jogo: Agente 1: {env.game.agent1_matches_won}, Agente 2: {env.game.agent2_matches_won}")
            print(f"Mao ganhas: Agente 1: {env.game.current_match.current_round.agent1_hands_won}, Agente 2: {env.game.current_match.current_round.agent2_hands_won}")
            print(f"Cartas Jogadas na Mão atual: {info['cards played in hand']}")
            print(f"Manilha: {info['manilha']}")

            action = -1
            if current_player == 2: # Jogador Humano
                print(f"Suas cartas: {info['your cards']}")
                print("Ações disponíveis: 0, 1, 2 (jogar carta), 3 (truco), 4 (passar), 5 (aumentar truco)")
                
                valid_input = False
                while not valid_input:
                    try:
                        action_input = input("Escolha sua ação (0-5): ")
                        action = int(action_input)
                        if 0 <= action <= 5:
                            valid_input = True
                        else:
                            print("Ação inválida. Por favor, escolha um número entre 0 e 5.")
                    except ValueError:
                        print("Entrada inválida. Por favor, digite um número.")
            else:
                print(f"Suas cartas: {info['your cards']}")
                print("Ações disponíveis: 0, 1, 2 (jogar carta), 3 (truco), 4 (correr), 5 (aumentar truco), 6 (aceitar truco), 7 (aceitar aumento), 8 (passar aumento)")
                
                valid_input = False
                while not valid_input:
                    try:
                        action_input = input("Escolha sua ação (0-8): ")
                        action = int(action_input)
                        if 0 <= action <= 8:
                            valid_input = True
                        else:
                            print("Ação inválida. Por favor, escolha um número entre 0 e 5.")
                    except ValueError:
                        print("Entrada inválida. Por favor, digite um número.")

            # Executa a ação no ambiente
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            print(f"  Recompensa para {player_name}: {reward}")
            print(f"  Jogo Terminado: {'Sim' if terminated else 'Não'}, Truncado: {'Sim' if truncated else 'Não'}")

            if done:
                game_winner = env.game.get_game_winner()
                print("\n--- FIM DO JOGO ---")
                if game_winner == 1:
                    print(f"!!! Agente 1 venceu o JOGO !!!")
                elif game_winner == 2:
                    print(f"!!! VOCÊ (Agente 2) venceu o JOGO !!!")
                else:
                    print(f"Jogo terminou em empate ou sem vencedor claro.")
                print(f"Placar Final do Jogo: Agente 1: {env.game.agent1_matches_won} partidas, Agente 2: {env.game.agent2_matches_won} partidas")
                
                play_again = input("Deseja jogar novamente? (s/n): ").lower()
                if play_again == 's':
                    break # Sai do loop interno do jogo, o loop externo continua para um novo jogo
                else:
                    return # Sai da função play_game

if __name__ == "__main__":
    play_game()

