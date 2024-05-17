class State:
    def __init__(self, position, reward, terminal) -> None:
        self.position = position
        self.reward = reward
        self.terminal = terminal

    def __str__(self) -> str:
        return f"position = {self.position}, reward = {self.reward}, terminal = {self.terminal}"
  
