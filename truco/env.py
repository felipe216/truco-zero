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
        self.action_space = spaces.Discrete(10)

        self.game = None
        self.reset(current_player=current_player)


    def reset(self, seed = None, options = None, current_player=1):
        super().reset(seed=seed, options=options)

        self.game = TrucoGame()
        self.game.star_new_match()
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
        hand_played = False

        current_match = self.game.current_match
        current_round = current_match.current_round

        if self.done:
            return self._get_observation(self.current_player), reward, True, truncated, self._get_info(self.current_player)

        #just play a card
        if action < 3:
            if self.mode == 'train':
                moves = [i for i in range(len(current_round.agent2_cards))]
                agent_move = self.agent.choose_action(moves)
                is_valid_move = current_round.validade_move(action, 1)
                if not is_valid_move:
                    reward += -current_match.truco_value
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)

                current_round.play_hand(action, agent_move)
                hand_played = True
            else:
                if current_match.truco_called:
                    current_match.accept_truco(self.current_player)
                    reward += 0.5
                    
                current_round.play_single_card(self.current_player, action)
                reward += 0.5

        elif action in [3, 4, 5]:
            if self.mode == 'train':
                is_valid_move = current_round.validade_move(action-3, 1)
                if not is_valid_move:
                    reward += -current_match.truco_value
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)
                if not current_match.check_truco(1):
                    reward += -0.5
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)
                reward += 0.5
                current_match.call_truco(1)
                moves = []
                for move in range(len(current_round.agent2_cards)):
                    moves.append(move+3)
                    moves.append(move+6)
                moves.append(9)
                agent_move = self.agent.choose_action(moves)
                if agent_move in [3, 4, 5]:
                    current_match.accept_truco(2)
                    agent_move -= 3
                elif agent_move in [6, 7, 8]:
                    current_match.raise_truco(2)
                    agent_move -= 6
                else:
                    current_match.fold_truco(2)
                current_round.play_hand(action-3, agent_move)
                hand_played = True
            else:
                if not current_match.check_truco(self.current_player):
                    reward += -0.5
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)
                reward += 0.5
                current_match.call_truco(self.current_player)
                

        elif action in [6, 7, 8]:
            if self.mode == 'train':
                is_valid_move = current_round.validade_move(action-6, 1)
                if not is_valid_move:
                    reward += -current_match.truco_value
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)
                valid_raise = current_match.raise_truco(1)
                if not valid_raise:
                    reward += -0.5
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)
                reward += 0.5
                current_match.accept_raise(2)
                moves = []
                for move in range(len(current_round.agent1_cards)):
                    moves.append(move+3)
                    moves.append(move+6)
                moves.append(9)
                agent_move = self.agent.choose_action(moves)
                if agent_move in [3, 4, 5]:
                    current_match.accept_truco(2)
                    agent_move -= 3
                    current_round.play_hand(action-6, agent_move)
                    hand_played = True
                elif agent_move in [6, 7, 8]:
                    current_match.raise_truco(2)
                    agent_move -= 6
                    current_round.play_hand(action-6, agent_move)
                    hand_played = True
                elif agent_move == 9:
                    current_match.fold_truco(2)
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
            else:
                valid_raise = current_match.raise_truco(self.current_player)
                if not valid_raise:
                    reward += -0.5
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)
                reward += 0.5
                current_match.accept_raise(self.current_player)
        elif action == 9:
            if self.mode == 'train':
                valid_fold = current_match.fold_truco(1)
                if not valid_fold:
                    reward += -0.5
                    current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
                    return self._get_observation(self.current_player), reward, terminated, truncated, self._get_info(self.current_player)
                reward += 0.5
            else:
                current_match.fold_truco(self.current_player)
                current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)

                

        
        if current_round.hand_ready():
            current_round.play_hand(current_round.card_agent1, current_round.card_agent2)
            hand_played = True

        if hand_played:
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
                current_match.start_new_round(current_round.agent1_cards, current_round.agent2_cards, current_round.manilha, current_round.cards_strength)
            
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
