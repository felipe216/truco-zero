
from truco.rules import get_round_winner

class TrucoRound:
    def __init__(self, agent1_cards, agent2_cards, manilha, cards_strength):
        self.agent1_cards = agent1_cards
        self.agent2_cards = agent2_cards
        self.manilha = manilha
        self.cards_strength = cards_strength
        self.cards_played = []
        self.agent1_hands_won = 0
        self.agent2_hands_won = 0
        self.current_hand_number = 1
        self.draw_in_hand = False
        self.last_card_agent1 = None
        self.last_card_agent2 = None
        self.cards_played_in_hand = []

    def play_single_card(self, agent_id, card_idx):
        if agent_id == 1:
            try:
                self.last_card_agent1 = self.agent1_cards.pop(card_idx)
                self.cards_played.append(self.last_card_agent1)
                self.cards_played_in_hand.append(self.last_card_agent1)
                return 1
            except:
                return -1
        elif agent_id == 2:
            try:
                self.last_card_agent2 = self.agent2_cards.pop(card_idx)
                self.cards_played.append(self.last_card_agent2)
                self.cards_played_in_hand.append(self.last_card_agent2)
                return 1
            except:
                return -1
        else:
            return -1

    def hand_ready(self):
        return self.card_agent1 and self.card_agent2

    def play_hand(self, agent1_card_idx, agent2_card_idx):
        self.cards_played.append((agent1_card_idx, agent2_card_idx))
        self.last_card_agent1 = self.agent1_cards.pop(agent1_card_idx)
        self.last_card_agent2 = self.agent2_cards.pop(agent2_card_idx)

        self.current_hand_number += 1

        winner = self.get_hand_winner()

        self.update_round_score(winner)
        return self.is_round_over()


    def get_hand_winner(self):
        return get_round_winner(self.last_card_agent1, self.last_card_agent2, self.cards_strength)
        
    def update_round_score(self, agent_id):
        if agent_id == 1:
            self.agent1_hands_won += 1
        elif agent_id == 2:
            self.agent2_hands_won += 1
        else:
            self.draw_in_hand = True

    def is_round_over(self):
        if self.draw_in_hand:
            return self.agent1_hands_won == 1 or self.agent2_hands_won == 1
        else:
            return self.agent1_hands_won == 2 or self.agent2_hands_won == 2

    def validade_move(self, agent_card_idx, agent_id):
        if agent_id == 1:
            try:
                self.agent1_cards[agent_card_idx]
            except:
                return False
        else:
            try:
                self.agent2_cards[agent_card_idx]
            except:
                return False
        
    def get_round_winner(self):
        if self.agent1_hands_won > self.agent2_hands_won:
            return 1
        else:
            return 2
        
    def both_cards_played_in_hand(self):
        return self.last_card_agent2 and self.last_card_agent1


class TrucoGame:
    def __init__(self, target_matches_to_win=12):
        self.agent1_matches_won = 0
        self.agent2_matches_won = 0
        self.target_matches_to_win = target_matches_to_win
        self.current_match = None


    def start_new_match(self):
        self.current_match = TrucoMatch(target_points_to_win=12)
    
    def update_game_score(self, winner_agent_id):
        if winner_agent_id == 1:
            self.agent1_matches_won += 1
        else:
            self.agent2_matches_won += 1

        return self.is_game_over()

    def is_game_over(self):
        return self.agent1_matches_won == self.target_matches_to_win or self.agent2_matches_won == self.target_matches_to_win
    
    def get_game_winner(self):
        if self.agent1_matches_won > self.agent2_matches_won:
            return 1
        else:
            return 2
    



class TrucoMatch:

    def __init__(self, truco_value=1, target_points_to_win=12):
        
        self.target_points_to_win = target_points_to_win
        self.agent1_score = 0
        self.agent2_score = 0
        self.truco_value = truco_value
        self.truco_called = False
        self.current_round = None
        self.player_truco = None
        self.raise_called = False
        self.player_raise = None
        self.truco_accepted = False
        self.raise_accepted = False


    def start_new_round(self, agent1_cards, agent2_cards, manilha, cards_strength):
        self.current_round = TrucoRound(agent1_cards, agent2_cards, manilha, cards_strength)
        self.truco_called = False
        self.truco_value = 1
        self.player_truco = None
    

    def update_match_score(self, winner_agent_id, points):
        if winner_agent_id == 1:
            self.agent1_score += points
        else:
            self.agent2_score += points

    
    def is_match_over(self):
        return self.agent1_score == self.target_points_to_win or self.agent2_score == self.target_points_to_win
    
    def get_match_winner(self):
        if self.agent1_score > self.agent2_score:
            return 1
        else:
            return 2
        
    def call_truco(self, player_id):
        self.truco_called = True
        self.player_truco = player_id
        self.truco_value = 3

    def check_truco(self, player_id):
        if player_id == self.player_truco:
            self.truco_called = False
            return False
        if self.truco_called or self.truco_accepted:
            return False
        else:
            return True

    def accept_truco(self, player_id):
        if player_id == self.player_truco:
            self.truco_called = False
            return False
        elif not self.truco_called:
            return False
        else:
            self.truco_value = 3
            self.truco_accepted = True
            return True

    def raise_truco(self, player_id):
        if self.truco_value == 12:
            return False
        elif not self.truco_called:
            return False
        elif player_id == self.player_raise:
            self.truco_called = False
            return False
        else:
            if self.truco_value == 1:
                self.truco_value = 3
            
            self.truco_value += 3
            self.raise_called = True
            self.player_raise = player_id
            return True

        
    def accept_raise(self, player_id):
        if self.truco_value == 1:
            self.truco_value = 3
        elif player_id == self.player_raise:
            self.truco_called = False
            self.raise_called = False
            return False
        else:
            if self.truco_value == 3:
                self.truco_value += 3
            self.raise_accepted = True
            return True

    def fold_truco(self, player_id):
        if player_id == self.player_truco or not self.truco_called:
            self.truco_called = False
            return False
        else:
            self.truco_value = 1
            self.truco_called = False
            self.truco_accepted = False
            self.raise_accepted = False
            self.player_raise = None
            self.raise_called = False
            self.player_truco = None
            if player_id == 1:
                self.agent2_score += self.truco_value
            else:
                self.agent1_score += self.truco_value
            return True
        
    def fold_raise(self, player_id):
        if player_id == self.player_raise:
            self.truco_called = False
            self.raise_called = False
            return False
        elif not self.raise_called:
            return False
        else:
            self.truco_value -= 3
            self.raise_called = False
            if player_id == 1:
                self.agent2_score += self.truco_value
            else:
                self.agent1_score += self.truco_value
            self.truco_accepted = False
            self.raise_called = False
            self.player_raise = None
            self.raise_accepted = False
            return True