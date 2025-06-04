import gymnasium as gym
from gymnasium import spaces
import numpy as np

from truco.rules import shuffle_and_deal, get_round_winner, get_card_strength, get_strongest_card, set_card_strength


class TrucoEnv(gym.Env):

    def __init__(self):
        super().__init__()

        
        self.observation_space = spaces.Box(low=0, high=20, shape=(8,), dtype=np.int32)
        
        self.agent1_matches_won = 0
        self.agent2_matches_won = 0

        # Define the observation and action spaces: 0, 1, 2: play card; 3 = truca, 4 = pass, 5 = raise
        self.action_space = spaces.Discrete(6)

        self.reset()


    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.agent1_cards, self.agent2_cards, self.manilha, self.cards, self.cards_strength = shuffle_and_deal()

        self.cards_strength = set_card_strength(self.cards_strength, self.manilha)

        self.cards_played = []
        self.oponent_cards_played = []

        self.rodada = 1
        self.truco_value = 1
        self.truco_called = False
        self.done = False
        self.agent1_score = 0
        self.agent2_score = 0
        self.draw = False
        self.agent1_rounds = 0
        self.agent2_rounds = 0

        return self._get_observation(), {}


    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if action == 3 and not self.truco_called:
            self.truco_called = True
            self.truco_value = 3
        elif action == 4 and not self.truco_called:
            reward = -self.truco_value
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info
        elif action == 5 and self.truco_called and self.truco_value != 12:
            self.truco_called = True
            self.truco_value += 3
        else:
            carta_idx = action
            if carta_idx >= len(self.agent1_cards):
                # invalid action
                reward = -self.truco_value
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, info
            
            agent1_card = self.agent1_cards.pop(carta_idx)
            agent2_card = get_strongest_card(self.cards_strength, self.agent2_cards)
            
            self.cards_played.append(agent1_card)
            self.oponent_cards_played.append(agent2_card)
            
            winner = get_round_winner(agent1_card, agent2_card, self.cards_strength)

            if winner == 1 and not self.draw:
                self.agent1_score += 1
            elif winner == 2 and not self.draw:
                self.agent2_score += 1
            else:
                self.draw = True

            self.rodada += 1

            if self.agent1_score == 2:
                reward = self.truco_value
                self.agent1_rounds += 1
                return self._get_observation(), reward, terminated, truncated, info
            elif self.agent2_score == 2:
                reward = -self.truco_value
                self.agent2_rounds += 1
                return self._get_observation(), reward, terminated, truncated, info
            elif winner == 1 and self.draw:
                self.agent1_rounds += 1
                reward = self.truco_value
                self.draw = False
                return self._get_observation(), reward, terminated, truncated, info
            elif winner == 2 and self.draw:
                reward = -self.truco_value
                self.draw = False
                self.agent2_rounds += 1
                return self._get_observation(), reward, terminated, truncated, info
            

            if self.agent1_rounds == 12:
                self.agent1_matches_won += 1
                return self._get_observation(), reward, terminated, truncated, info
            elif self.agent2_rounds == 12:
                self.agent2_matches_won += 1
                return self._get_observation(), reward, terminated, truncated, info

            if self.agent1_matches_won == 2 or self.agent2_matches_won == 2:
                terminated = True
                self.done = True
                return self._get_observation(), reward, terminated, truncated, info

        return self._get_observation(), reward, terminated, truncated, info
            

    def _get_observation(self):
        obs = [
            self.rodada,
            self.truco_value,
            self.truco_value,
            int(self.truco_called),
            self.agent1_matches_won,
            self.agent2_matches_won,
            self.agent1_score,
            self.agent2_score
        ]

        return np.array(obs, dtype=np.int32)
            