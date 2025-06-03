import gymnasium as gym
from gymnasium import spaces
import numpy as np

from truco.rules import shuffle_and_deal, get_round_winner, get_card_strength, get_strongest_card, set_card_strength


class TrucoEnv(gym.Env):

    def __init__(self):
        super().__init__()

        
        self.observation_space = spaces.Box(low=0, high=20, shape=(6,0), dtype=np.float32)
        
        # Define the observation and action spaces: 0, 1, 2: play card; 3 = truca, 4 = pass, 5 = raise
        self.action_space = spaces.Discrete(6)

        self.reset()


    def reset(self, seed = None, options = None):
        super().reset(seed=seed, options=options)

        self.agent1_cards, self.agent2_cards, self.manilha, self.cards, self.cards_strength = shuffle_and_deal()

        self.cards_strength = set_card_strength(self.cards, self.manilha)

        self.cards_played = []
        self.oponent_cards_played = []

        self.rodada = 1
        self.truco_value = 1
        self.truco_called = False
        self.done = False
        self.agent1_score = 0
        self.agent2_score = 0
        self.draw = False

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
            if carta_idx >= len(self.mao_agente):
                # invalid action
                reward = -self.valor_truco
                terminated = True
                self.done = True
                return self._get_obs(), reward, terminated, truncated, info
            
            agent1_card = self.agent1_cards.pop(carta_idx)
            agent2_card = get_strongest_card(self.cards_strength, self.agent2_cards)

            winner = get_round_winner(agent1_card, agent2_card)

            if winner == 1 and not self.draw:
                self.agent1_score += 1
            elif winner == 2 and not self.draw:
                self.agent2_score += 1
            else:
                self.draw = True

            self.rodada += 1

            if self.agent1_score == 2:
                reward = self.truco_value
                terminated = True
                self.done = True
                return self._get_obs(), reward, terminated, truncated, info
            elif self.agent2_score == 2:
                reward = -self.truco_value
                terminated = True
                self.done = True            
                return self._get_obs(), reward, terminated, truncated, info
            elif winner == 1 and self.draw:
                reward = self.truco_value
                terminated = True
                self.done = True
                self.draw = False
                return self._get_obs(), reward, terminated, truncated, info
            elif winner == 2 and self.draw:
                reward = -self.truco_value
                terminated = True
                self.done = True
                self.draw = False
                return self._get_obs(), reward, terminated, truncated, info
            

    def _get_observation(self):
        obs = [
            self.cards[self.agent1_cards[i]] if i < len(self.agent1_cards) else 0
            for i in range(3)
        ]
        obs.append(self.rodada)
        obs.append(self.truco_value)
        return np.array(obs, dtype=np.int32)
            