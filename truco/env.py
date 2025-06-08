import gymnasium as gym
from gymnasium import spaces
import numpy as np

from truco.rules import shuffle_and_deal, get_round_winner, get_card_strength, get_strongest_card, set_card_strength
from truco.game_logic import TrucoGame
from agents.random_agent import RandomAgent


class TrucoEnv(gym.Env):

    def __init__(self, mode='train'):
        super().__init__()

        self.mode = mode
        self.observation_space = spaces.Box(low=0, high=20, shape=(10,), dtype=np.int32)
        
        # Define the observation and action spaces: [0, 1, 2]: play card; [3, 4, 5]: truco; [6, 7, 8]: raise, 9: pass
        # can pass more than one action: play a card and trucar, or play a card and raise
        self.action_space = spaces.Discrete(10)

        self.game = None
        self.reset()


    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.game = TrucoGame()
        self.game.star_new_match()

        agent1_cards, agent2_cards, manilha, cards, cards_strength = shuffle_and_deal()

        cards_strength = set_card_strength(cards_strength, manilha)
        self.game.current_match.start_new_round(agent1_cards, agent2_cards, manilha, cards_strength)

        self.done = False
        self.agent = RandomAgent(name='Random Agent')
        return self._get_observation(), {}


    def step(self, action, current_player=1):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        self.current_player = current_player
        current_match = self.game.current_match
        current_round = current_match.current_round

        if self.done:
            return self._get_observation(self.current_player), reward, True, truncated, info

        #just play a card
        if action < 3:
            if self.mode == 'train':
                moves = [i for i in range(len(current_round.agent2_cards))]
                agent_move = self.agent.choose_action(moves)
                is_valid_move = current_round.validade_move(action, 1)
                if not is_valid_move:
                    reward += -current_match.truco_value
                    current_match.star_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, info

                current_round.play_hand(action, agent_move)

        elif action in [3, 4, 5]:
            if self.mode == 'train':
                if not current_match.check_truco(1):
                    reward += -0.5
                    current_match.star_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(), reward, terminated, truncated, info
                reward += 0.5
                current_match.call_truco(1)
                moves = []
                for move in range(len(current_round.agent2_cards)):
                    moves.append(move+3)
                    moves.append(move+6)
                moves.append(9)
                agent_move = self.agent.choose_action(moves)
                if agent_move < 6:
                    current_match.accept_truco(2)
                    agent_move -= 3
                    current_round.play_hand(agent_move-3, agent_move)
                elif agent_move < 9:
                    current_match.raise_truco(2)
                    agent_move -= 6
                    return self._get_observation(self.current_player), reward, terminated, truncated, info

        elif action in [6, 7, 8]:
            if self.mode == 'train':
                valid_raise = current_match.raise_truco(1)
                if not valid_raise:
                    reward += -0.5
                    current_match.star_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, info
                reward += 0.5
                current_match.accept_raise(1)
                moves = []
                for move in range(len(current_round.agent1_cards)):
                    moves.append(move+3)
                    moves.append(move+6)
                moves.append(9)
                agent_move = self.agent.choose_action(moves)
                if agent_move < 6:
                    current_match.accept_truco(1)
                    agent_move -= 3
                    current_round.play_hand(agent_move-3, agent_move)
                elif agent_move < 9:
                    current_match.raise_truco(1)

            

        winner = current_round.get_hand_winner()
        if winner == 1:
            reward += 1
        else:
            reward += -1
        is_round_over = current_round.is_round_over()
        if is_round_over:
            winner = current_round.get_round_winner()
            current_match.update_match_score(winner, current_match.truco_value)
            winner = current_round.get_round_winner()
            if winner == 1:
                reward += current_match.truco_value
            else:
                reward += -current_match.truco_value
            current_match.star_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
        
        if current_match.is_match_over():
            winner = current_match.get_match_winner()
            if winner == 1:
                reward += 10
            else:
                reward += -10
            is_game_done = self.game.update_game_score(winner)
            if is_game_done:
                terminated = True
                self.done = True
                winner = self.game.get_game_winner()
                if winner == 1:
                    reward += 100
                else:
                    reward += -100
        
        
        
        
        return self._get_observation(self.current_player), reward, terminated, truncated, info
                
                        

    def _get_observation(self):
        obs = [
            self.truco_value,
            int(self.truco_called),
            self.agent1_matches_won,
            self.agent2_matches_won,
            self.agent1_score,
            self.agent2_score
        ]
        i = 0
        for card in self.agent1_cards:
            obs.append(get_card_strength(card, self.cards_strength))
            i += 1
        obs.extend([-1] * (3 - len(self.agent1_cards)))

        return np.array(obs, dtype=np.int32)
    
    def _get_human_observation(self):
        obs = {
            "rodada": self.rodada,
            "truco_value": self.truco_value,
            "your cards": self.agent2_cards,
            "trucado": self.truco_called,
            "your score": self.agent1_score,
            "oponent score": self.agent2_score,
            "agent1_matches_won": self.agent1_matches_won,
            "agent2_matches_won": self.agent2_matches_won,
            "agent_1_rounds": self.agent1_rounds,
            "agent_2_rounds": self.agent2_rounds,
            "bot_cards": self.agent1_cards
        }

        return obs