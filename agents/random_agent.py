import random

class RandomAgent:

    def __init__(self, name="Random Agent"):
        self.name = name
        pass

    def choose_action(self, available_actions):
        return random.choice(available_actions)