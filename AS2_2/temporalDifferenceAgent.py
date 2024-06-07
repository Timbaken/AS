from policy import Policy
from agent import Agent
from maze import Maze
from state import State

import random
from tqdm import tqdm
from tabulate import tabulate

class TemporalDifferenceAgent(Agent):
    def __init__(self, maze: Maze, policy: Policy) -> None:
        super().__init__(maze, policy)

        self.values = dict()

    def temporal_difference(self, 
                            alpha: float=.1, 
                            gamma: float=1,
                            iterations: int=100,
                            visualize: bool=True
                            ) -> None:
        for row in self.maze.states:
            for state in row:
                if state.terminal:
                    self.values[state] = 0
                else:
                    self.values[state] = random.randint(0, 10)

        start = self.currentposition
        
        for _ in tqdm(range(iterations)):
            current_state: State = self.maze.states[self.currentposition]
            
            while not current_state.terminal:
                self.act()
                reward = self.maze.rewards[self.currentposition]
                new_state = self.maze.states[self.currentposition]
                
                self.values[current_state] += alpha * (
                    reward +
                    gamma *  
                    self.values[new_state] -
                    self.values[current_state]
                )

                current_state = new_state
        
            self.currentposition = start
        
        if visualize:
            grid = [
                [
                    self.values[state] for state in row
                ] for row in self.maze.states
            ]
            print(tabulate(grid, tablefmt="grid"))