import random



TRUCO_CARDS = [
    '4E', '5E', '6E', '7E', 'QE', 'JE', 'KE', 'AE', '2E','3E',
    '4C', '5C', '6C', '7C', 'QC', 'JC', 'KE', 'AC', '2C', '3C',
    '4O', '5O', '6O', '7O', 'QO', 'JO', 'KO', 'AO', '2O', '3O',
    '4P', '5P', '6P', '7P', 'QP', 'JP', 'KP', 'AP', '2P', '3P'
]

CARDS_STRENGTH = {
    '4E': 1, '5E': 2, '6E': 3, '7E': 4, 'QE': 5, 'JE': 6, 'KE': 7, 'AE': 8, '2E': 9,'3E': 10,
    '4C': 1, '5C': 2, '6C': 3, '7C': 4, 'QC': 5, 'JC': 6, 'KE': 7, 'AC': 8, '2C': 9, '3C': 10,
    '4O': 1, '5O': 2, '6O': 3, '7O': 4, 'QO': 5, 'JO': 6, 'KO': 7, 'AO': 8, '2O': 9, '3O': 10,
    '4P': 1, '5P': 2, '6P': 3, '7P': 4, 'QP': 5, 'JP': 6, 'KP': 7, 'AP': 8, '2P': 9, '3P': 10
}

def shuffle_and_deal():
    cards = TRUCO_CARDS[:]
    random.shuffle(TRUCO_CARDS)
    cards = random.sample(cards, 7)
    return cards[:3], cards[3:6], cards[6], cards, CARDS_STRENGTH


def get_card_strength(card: str, cards: dict) -> int:
    return cards[card]

def get_round_winner(player1_card: str, player2_card: str, cards_strength: dict) -> int:
    if get_card_strength(player1_card, cards_strength) > get_card_strength(player2_card, cards_strength):
        return 1
    elif get_card_strength(player1_card, cards_strength) < get_card_strength(player2_card, cards_strength):
        return 2
    else:
        return 0


def set_card_strength(cards: dict, manilha: str) -> dict:
    card_number = manilha[0]

    suits = ['E', 'C', 'O', 'P']
    try: 
        card_number_int = int(card_number)
        is_number = True
    except: 
        is_number = False

    if is_number:
        if card_number_int < 7:
            for suit in suits:
                cards[f'{card_number_int+1}{suit}'] = 11
        else:
            for suit in suits:
                cards[f'Q{suit}'] = 11

    else:
        if card_number == 'A':
            for suit in suits:
                cards[f'2{suit}'] = 11
        elif card_number == 'K':
            for suit in suits:
                cards[f'A{suit}'] = 11
        elif card_number == 'J':
            for suit in suits:
                cards[f'K{suit}'] = 11
        elif card_number == 'Q':
            for suit in suits:
                cards[f'J{suit}'] = 11

    return cards



def get_strongest_card(cards: list, player_cards: list) -> str:
    greater_card = cards[player_cards[0]]
    greater_card_key = player_cards[0]
    for player_card in player_cards:
        if cards[player_card] > greater_card:
            greater_card = cards[player_card]
            greater_card_key = player_card

    return greater_card_key

if __name__ == "__main__":
    cards = set_card_strength(CARDS_STRENGTH, '4E')
    print(cards)