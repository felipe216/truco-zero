import gymnasium as gym
from gymnasium import spaces
import numpy as np

from truco.rules import shuffle_and_deal, get_round_winner, get_card_strength, get_strongest_card, set_card_strength


class TrucoEnv(gym.Env):

    def __init__(self, mode='train'):
        super().__init__()

        self.mode = mode
        self.observation_space = spaces.Box(low=0, high=20, shape=(10,), dtype=np.int32)
        
        self.agent1_matches_won = 0
        self.agent2_matches_won = 0

        self.agent1_rounds = 0
        self.agent2_rounds = 0
        self.agent1_score = 0
        self.agent2_score = 0

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
        self.draw = False
        self.last_card_agent1 = None
        self.last_card_agent2 = None

        return self._get_observation(), {"round": self.rodada, "manilha": self.manilha, "your cards": self.agent2_cards, "your score": self.agent1_score, "oponent score": self.agent2_score,
            "trucado": self.truco_called, "truco value": self.truco_value, "cards played": self.cards_played, "oponent cards played": self.oponent_cards_played, "agent_1_matches": self.agent1_matches_won, "agent_2_matches": self.agent2_matches_won,
            "agent_1_rounds": self.agent1_rounds, "agent_2_rounds": self.agent2_rounds, "bot_cards": self.agent1_cards}


    def step(self, action, player_turn=None, second_action=None):
        reward = 0
        terminated = False
        truncated = False
        info = {}
 
        if self.rodada > 3:
            terminated = True
            self.done = True

        if self.done:
            return self._get_observation(), reward, True, truncated, info

        if action == 3:
            if self.truco_called:
                self.truco_called = True
                self.truco_value = 3
            else:
                reward = -self.truco_value
                terminated = True
                self.done = True
        elif action == 4:
            reward = -self.truco_value
            terminated = True
            self.done = True
            return self._get_observation(), reward, terminated, truncated, info
        elif action == 5:
            if self.truco_called and self.truco_value != 12:
                self.truco_called = True
                self.truco_value += 3
            else:
                reward = -self.truco_value
                terminated = True
                self.done = True
        else:
            if self.mode == 'train':
                self._make_agent1_move(action)
                agent2_card = get_strongest_card(self.cards_strength, self.agent2_cards)
                self._make_agent2_move(agent2_card)
                reward = self._check_round_winner()
            elif self.mode == 'game':
                if player_turn == 1:
                    self._make_agent1_move(action)
                elif player_turn == 2:
                    self._make_agent2_move(action)

                if second_action is not None:
                    reward = self._check_round_winner()
            
            if self._check_end_game():
                terminated = True
                self.done = True
            
            

        return self._get_observation(), reward, terminated, truncated, info
            

    def _get_observation(self):
        obs = [
            self.rodada,
            self.truco_value,
            int(self.truco_called),
            self.agent1_matches_won,
            self.agent2_matches_won,
            self.agent1_score,
            self.agent2_score
        ]
        
        for card in self.agent1_cards:
            print(card)
            obs.append(get_card_strength(card, self.cards_strength))

        return np.array(obs, dtype=np.int32)
    
    def _get_human_observation(self):
        obs = {
            "rodada": self.rodada,
            "truco_value": self.truco_value,
            "your cards": self.agent2_cards,
            "trucado": self.truco_called,
            "your score": self.agent1_score,
            "oponent score": self.agent2_score,
            "agent_1_matches": self.agent1_matches_won,
            "agent_2_matches": self.agent2_matches_won,
            "agent_1_rounds": self.agent1_rounds,
            "agent_2_rounds": self.agent2_rounds,
            "bot_cards": self.agent1_cards
        }

        return obs
    

    def _make_agent1_move(self, card_idx: int):
        if card_idx >= len(self.agent1_cards):
            return
            # invalid action
        agent1_card = self.agent1_cards.pop(card_idx)
            
        self.cards_played.append(agent1_card)
        self.last_card_agent1 = agent1_card

    def _make_agent2_move(self, card_idx: str):
        if card_idx not in self.agent2_cards:
            return
            # invalid action
        agent2_card = card_idx
        self.agent2_cards.remove(card_idx)
            
        self.cards_played.append(agent2_card)
        self.last_card_agent2 = agent2_card

    def _check_round_winner(self):
        winner = get_round_winner(self.last_card_agent1, self.last_card_agent2, self.cards_strength)

        
        if winner == 1 and not self.draw:
            self.agent1_score += 1
        elif winner == 2 and not self.draw:
            self.agent2_score += 1
        else:
            self.draw = True

        self.rodada += 1

        if self.agent1_score == 2:
            self.agent1_rounds += 1
            return self.truco_value
        elif self.agent2_score == 2:
            self.agent2_rounds += 1
            return -self.truco_value
        elif winner == 1 and self.draw:
            self.agent1_rounds += 1
            self.draw = False
            return self.truco_value 
        elif winner == 2 and self.draw:
            self.draw = False
            self.agent2_rounds += 1
            return -self.truco_value

        return 0
    

    def _check_end_game(self):
        if self.agent1_rounds >= 12:
            self.agent1_matches_won += 1
            self.agent1_score = 0
            self.agent2_score = 0
            return True
        elif self.agent2_rounds >= 12:
            self.agent2_matches_won += 1
            self.agent1_score = 0
            self.agent2_score = 0
            return True

        if self.agent1_matches_won == 2 or self.agent2_matches_won == 2:
            return True

        return False