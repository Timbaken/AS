from policy import Policy
from state import State

class OptimalPolicy(Policy):
    def __init__(self, actions: dict[State: str]) -> None:
        super().__init__()
        
        self.actions = actions

    def select_action(self, state: State):
        return self.actions[state]
    
    def __str__(self) -> str:
        return self.__class__.__name__