import gymnasium as gym
from gymnasium import spaces
import numpy as np

from truco.rules import shuffle_and_deal, get_round_winner, get_card_strength, get_strongest_card, set_card_strength
from truco.game_logic import TrucoGame
from agents.random_agent import RandomAgent


class TrucoEnv(gym.Env):

    def __init__(self, mode='train', current_player=1):
        super().__init__()

        self.mode = mode
        self.observation_space = spaces.Box(low=-30, high=300, shape=(15,), dtype=np.int32)
        
        # Define the observation and action spaces: [0, 1, 2]: play card; [3, 4, 5]: truco; [6, 7, 8]: raise, 9: pass
        # can pass more than one action: play a card and trucar, or play a card and raise
        self.action_space = spaces.Discrete(9)

        self.game = None
        self.reset(current_player=current_player)


    def reset(self, seed = None, options = None, current_player=1):
        super().reset(seed=seed, options=options)

        self.game = TrucoGame()
        self.game.start_new_match()
        self.current_player = current_player

        agent1_cards, agent2_cards, manilha, cards, cards_strength = shuffle_and_deal()

        cards_strength = set_card_strength(cards_strength, manilha)
        self.game.current_match.start_new_round(agent1_cards, agent2_cards, manilha, cards_strength)

        self.done = False
        self.agent = RandomAgent(name='Random Agent')
        return self._get_observation(self.current_player), self._get_info(self.current_player)


    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = self._get_info(self.current_player)

        current_match = self.game.current_match
        current_round = current_match.current_round

        if self.done:
            return self._get_observation(self.current_player), reward, True, truncated, info

        acting_player = self.current_player
        next_player = 3 - acting_player

        if action < 3:
            if (current_match.truco_called and not current_match.truco_accepted) or (current_match.raise_called and not current_match.raise_accepted):
                reward += -0.3
                result = -1
                terminated = True
                self.done = True
            else:
                result = current_round.play_single_card(acting_player, action)

            if result == -1:
                reward = -10
                terminated = True
                self.done = True

            else:
                reward += 0.3

                if current_round.both_cards_played_in_hand():
                    winner_hand = current_round.get_hand_winner()
                    current_round.update_round_score(winner_hand)
                    current_round.current_hand_number += 1
                    
                    if winner_hand == acting_player:
                        reward += 1
                    elif winner_hand != 0:
                        reward += -1

                    current_round.last_card_agent1 = None 
                    current_round.last_card_agent2 = None 
                    current_round.cards_played_in_hand = [] 

                    if current_round.is_round_over():
                        round_winner = current_round.get_round_winner()
                        current_match.update_match_score(round_winner, current_match.truco_value)
                        
                        if round_winner == acting_player:
                            reward += current_match.truco_value
                        elif round_winner != 0:
                            reward += -current_match.truco_value

                        agent1_cards, agent2_cards, manilha, cards, cards_strength = shuffle_and_deal()
                        cards_strength = set_card_strength(cards_strength, manilha)
                        self.game.current_match.start_new_round(agent1_cards, agent2_cards, manilha, cards_strength)

                        if current_match.is_match_over():
                            match_winner = current_match.get_match_winner()
                            self.game.update_game_score(match_winner)
                            
                            if match_winner == acting_player:
                                reward += 10
                            else:
                                reward += -10

                            if self.game.is_game_over():
                                terminated = True
                                self.done = True
                                game_winner = self.game.get_game_winner()

                                if game_winner == acting_player:
                                    reward += 100
                                else:
                                    reward += -100
                            else:
                                self.game.start_new_match()
                                agent1_cards, agent2_cards, manilha, cards, cards_strength = shuffle_and_deal()
                                cards_strength = set_card_strength(cards_strength, manilha)
                                self.game.current_match.start_new_round(agent1_cards, agent2_cards, manilha, cards_strength)

                    if not terminated:
                        self.current_player = next_player
                else:
                    self.current_player = next_player

        elif action == 3:
            if current_match.check_truco(acting_player):
                current_match.call_truco(acting_player)
                reward += 0.5
                self.current_player = next_player
            else:
                reward = -5
                terminated = True
                self.done = True

        elif action == 4:
            if current_match.fold_truco(acting_player):
                reward += 0.5
                reward = -current_match.truco_value
                terminated = True
                self.done = True
            else:
                reward = -5
                terminated = True
                self.done = True

        elif action == 5:
            if current_match.raise_truco(acting_player):
                reward += 0.5
                self.current_player = next_player
            else:
                reward = -5
                terminated = True
                self.done = True
        elif action == 6:
            if current_match.accept_truco(acting_player):
                reward += 0.5
                self.current_player = next_player
            else:
                reward = -5
                terminated = True
                self.done = True
        elif action == 7: # accept raise
            if current_match.accept_raise(acting_player):
                reward += 0.5
                self.current_player = next_player
            else:
                reward = -5
                terminated = True
                self.done = True
        elif action == 8: # fold raise
            current_match.fold_raise(acting_player)
            reward = -current_match.truco_value
            terminated = True
            self.done = True
        # --- Lógica para Aceitar/Recusar Truco/Aumento (Se as ações 6,7,8,9 forem para isso) ---
        # Se as ações 6, 7, 8, 9 são para aceitar/recusar truco/aumento, elas devem ser tratadas aqui.
        # É crucial que o ambiente saiba qual ação de truco está pendente (chamada ou aumento).
        # Exemplo (apenas para ilustrar, pode precisar de mais estado no TrucoMatch):
        # elif action == 6: # Aceitar Truco/Aumento
        #     if current_match.accept_pending_truco(acting_player):
        #         reward += 0.2 # Recompensa por aceitar e continuar o jogo
        #         self.current_player = next_player # O jogo continua, próximo turno
        #     else:
        #         reward = -1 # Penalidade por aceitar truco inválido
        #         terminated = True
        #         self.done = True
        # elif action == 7: # Recusar Truco/Aumento (Desistir)
        #     current_match.fold_pending_truco(acting_player)
        #     reward = -current_match.truco_value # Penalidade por recusar
        #     terminated = True
        #     self.done = True

        # Retorna a observação para o current_player (que pode ter sido alternado)
        return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)
                
                        


    def _get_observation(self, player_id):
        current_match = self.game.current_match
        current_round = current_match.current_round

        if player_id == 1:
            agent_cards = current_round.agent1_cards
            agent_hands_won = current_round.agent1_hands_won
            opponent_hands_won = current_round.agent2_hands_won
            agent_score_match = current_match.agent1_score
            opponent_score_match = current_match.agent2_score
        else: # player_id == 2
            agent_cards = current_round.agent2_cards
            agent_hands_won = current_round.agent2_hands_won
            opponent_hands_won = current_round.agent1_hands_won
            agent_score_match = current_match.agent2_score
            opponent_score_match = current_match.agent1_score

        obs = [
            current_round.current_hand_number, # Current hand number (1-3)
            current_match.truco_value, # Current truco value
            int(current_match.truco_called), # Is truco called?
            self.game.agent1_matches_won, # Agent 1 matches won
            self.game.agent2_matches_won, # Agent 2 matches won
            agent_score_match, # Agent's score in current match
            opponent_score_match, # Opponent's score in current match
            agent_hands_won, # Agent's hands won in current round
            opponent_hands_won, # Opponent's hands won in current round
            get_card_strength(current_round.manilha, current_round.cards_strength) # Manilha strength
        ]
        
        # Add agent's cards strength to observation
        for card in agent_cards:
            obs.append(get_card_strength(card, current_round.cards_strength))
        obs.extend([-1] * (3 - len(agent_cards))) # Pad with -1 if less than 3 cards

        # Add cards played in current hand (from agent's perspective)
        # If agent 1 played first, cards_played_in_hand will have agent1's card first
        # If agent 2 played first, cards_played_in_hand will have agent2's card first
        # We need to ensure the observation reflects the order from the current player's perspective
        if len(current_round.cards_played_in_hand) == 1:
            if player_id == 1: # Agent 1 is observing, and Agent 2 is about to play
                obs.append(get_card_strength(current_round.cards_played_in_hand[0], current_round.cards_strength)) # Agent 1's card
                obs.append(-1) # Opponent's card not yet played
            else: # Agent 2 is observing, and Agent 1 is about to play
                obs.append(get_card_strength(current_round.cards_played_in_hand[0], current_round.cards_strength)) # Agent 2's card
                obs.append(-1) # Opponent's card not yet played
        elif len(current_round.cards_played_in_hand) == 2:
            if player_id == 1: # Agent 1 is observing, both played
                obs.append(get_card_strength(current_round.last_card_agent1, current_round.cards_strength))
                obs.append(get_card_strength(current_round.last_card_agent2, current_round.cards_strength))
            else: # Agent 2 is observing, both played
                obs.append(get_card_strength(current_round.last_card_agent2, current_round.cards_strength))
                obs.append(get_card_strength(current_round.last_card_agent1, current_round.cards_strength))
        else: # No cards played yet
            obs.extend([-1] * 2)

        return np.array(obs, dtype=np.int32)
    
    def _get_info(self, player_id):
        current_match = self.game.current_match
        current_round = current_match.current_round

        if player_id == 1:
            your_cards = current_round.agent1_cards
            opponent_cards = current_round.agent2_cards
            your_score = current_match.agent1_score
            opponent_score = current_match.agent2_score
        else:
            your_cards = current_round.agent2_cards
            opponent_cards = current_round.agent1_cards
            your_score = current_match.agent2_score
            opponent_score = current_match.agent1_score

        info = {
            "round": current_round.current_hand_number,
            "manilha": current_round.manilha,
            "your cards": your_cards,
            "your score": your_score,
            "oponent score": opponent_score,
            "trucado": current_match.truco_called,
            "truco value": current_match.truco_value,
            "cards played in hand": current_round.cards_played_in_hand,
            "agent_1_matches": self.game.agent1_matches_won,
            "agent_2_matches": self.game.agent2_matches_won,
            "agent_1_rounds": current_match.agent1_score, # This is actually match score
            "agent_2_rounds": current_match.agent2_score, # This is actually match score
            "bot_cards": opponent_cards # This should be opponent's cards
        }
        return info
