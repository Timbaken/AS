import numpy as np
from state import State
from termcolor import colored

import numpy as np

class Maze:
    def __init__(self, rewards: np.ndarray, terminalpositions: list[tuple[int, int]], startposition: tuple[int, int]) -> None:
        self.rewards = rewards
        self.terminalpositions = terminalpositions
        self.startPosition = startposition
        
        self.states = np.empty([len(self.rewards), len(self.rewards[0])], dtype=object) 

        for x in range(len(self.rewards)):
            for y in range(len(self.rewards[0])):
                if (x, y) in self.terminalpositions:
                    self.states[x,y] = State((x,y), self.rewards[x,y], True)
                else:
                    self.states[x,y] = State((x,y), self.rewards[x,y], False)

        self.actions = {
            "up": (-1, 0),
            "down": (1, 0),
            "right": (0, 1),
            "left": (0, -1)
        }

    def step(self, coordinates: tuple[int, int], action: str):
        newPosition = tuple(a + b for a, b in zip(coordinates, self.actions[action]))
        length = self.states.shape[0]

        if newPosition[0] >= 0 and newPosition[1] >= 0 and newPosition[0] < length and newPosition[1] < length:
            return newPosition
        else:
            return coordinates

    def __str__(self) -> str:
        maze = ""
        for x in range(len(self.rewards)):
            for y in range(len(self.rewards[0])):
                if (x, y) == self.startPosition:
                    maze += colored(self.rewards[x][y], "yellow")
                elif (x, y) in self.terminalpositions:
                    maze += colored(self.rewards[x][y], "green")
                elif self.rewards[x][y] < -5:
                    maze += colored(self.rewards[x][y], "red")
                else:
                    maze += str(self.rewards[x][y])
                maze += ", "

            maze += "\n"
        return maze

