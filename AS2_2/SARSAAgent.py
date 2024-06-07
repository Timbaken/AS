from agent import Agent
from maze import Maze
from state import State

import random
from tqdm import tqdm
from tabulate import tabulate
from termcolor import colored


class SARSAAgent(Agent):
    def __init__(self, maze: Maze) -> None:
        super().__init__(maze, None)

        self.q: dict[State: dict[str: float]] = {}

    def greedy(self, state_dict: dict[str: float], epsilon: float):
        if random.random() < epsilon:
            return random.choice(self.actions)
        else:
            return max(state_dict, key=state_dict.get)

    def SARSA(self,
              alpha: float=.1,
              epsilon: float=.1,
              gamma: float=1,
              iterations: int=100,
              visualize: bool=True
              ):
        
        for row in self.maze.states:
            for state in row:
                if state.terminal:
                    self.q[state] = {action: 0 for action in self.actions}
                else:
                    self.q[state] = {action: random.randint(0, 10) for action in self.actions}
        
        for _ in tqdm(range(iterations)):
            current_state: State = self.maze.states[self.currentposition]

            a = self.greedy(self.q[current_state], epsilon)

            while not current_state.terminal:
                position = self.maze.step(current_state.position, a)
                reward = self.maze.rewards[position]
                new_state = self.maze.states[position]
                
                new_a = self.greedy(self.q[new_state], epsilon)

                self.q[current_state][a] += alpha * (
                    reward + 
                    gamma *
                    self.q[new_state][new_a] -
                    self.q[current_state][a]
                )

                current_state = new_state
                a = new_a

        if visualize:
            self.visualize()

    def state_str(self, state):
        output = ""
        
        for action, value in self.q[state].items():
            if value == max(self.q[state].values()):
                output += colored(f"{action}: {round(value, 2)}", "green")
            else:
                output += colored(f"{action}: {round(value, 2)}", "white")
            output += "\n"
        return output.rstrip()

    def visualize(self):
        grid = [
            [
                self.state_str(state) for state in row
            ] for row in self.maze.states
        ]
        print(tabulate(grid, tablefmt="grid"))