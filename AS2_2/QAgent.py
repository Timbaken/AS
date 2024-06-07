from SARSAAgent import SARSAAgent
from maze import Maze
from state import State

import random
from tqdm import tqdm


class QAgent(SARSAAgent):
    def __init__(self, maze: Maze) -> None:
        super().__init__(maze)

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
                else:
                    self.q[state] = {action: random.randint(0, 10) for action in self.actions}
        
        for _ in tqdm(range(iterations)):
            current_state: State = self.maze.states[self.currentposition]

            while not current_state.terminal:
                a = self.greedy(current_state, epsilon)

                position = self.maze.step(current_state.position, a)
                reward = self.maze.rewards[position]
                new_state = self.maze.states[position]
                
                new_a = self.greedy(new_state, 0)

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