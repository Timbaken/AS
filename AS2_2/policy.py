import random

class Policy:
    def __init__(self) -> None:
        pass

    def select_action(self, state):
        index = random.randint(0, 3)
        return ["up", "down", "right", "left"][index]
    
    def __str__(self) -> str:
        return self.__class__.__name__
   
