from QAgent import QAgent
from maze import Maze
from state import State

import random
from tqdm import tqdm


class DoubleQAgent(QAgent):
    def __init__(self, maze: Maze) -> None:
        super().__init__(maze)

        self.q_two: dict[State: dict[str: float]] = {}

    def q_learning(self,
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
                    self.q_two[state] = {action: 0 for action in self.actions}
                else:
                    self.q[state] = {action: random.randint(0, 10) for action in self.actions}
                    self.q_two[state] = {action: random.randint(0, 10) for action in self.actions}
        
        for _ in tqdm(range(iterations)):
            current_state: State = self.maze.states[self.currentposition]

            while not current_state.terminal:
                a = self.greedy(
                    {
                        key: 
                        self.q[current_state][key] + 
                        self.q_two[current_state][key] 
                        for key in self.q[current_state]
                    }, epsilon
                )

                position = self.maze.step(current_state.position, a)
                reward = self.maze.rewards[position]
                new_state = self.maze.states[position]

                choice = random.choice([self.q, self.q_two])
                
                new_a = self.greedy(choice[new_state], 0)

                choice[current_state][a] += alpha * (
                    reward + 
                    gamma *
                    choice[new_state][new_a] -
                    choice[current_state][a]
                )

                current_state = new_state

        if visualize:
            self.visualize()