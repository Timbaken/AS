from policy import Policy
from maze import Maze
import numpy as np
from termcolor import colored


class Valueiterationpolicy(Policy):
    def __init__(self, maze: Maze, treshold: float, discount: float, visualize: bool) -> None:
        super().__init__()

        self.maze = maze
        self.actions = {state: "up" for state in self.maze.states.flatten()}
        self.values = {state: 0 for state in self.maze.states.flatten()}

        self.determinePolicy(treshold, discount, visualize)

    def valueIteration(self, treshold, discount, visualize):
        delta = float("inf")
        iteration = 0
        
        while delta > treshold:
            delta = 0
            newValues = self.values.copy()
            for state in self.maze.states.flatten():
                if state.terminal:
                    newValues[state] = 0
                    continue
                newValues[state] = max([self.maze.states[self.maze.step(state.position, direction)].reward + discount * self.values[self.maze.states[self.maze.step(state.position, direction)]] 
                    for direction in ["up", "down", "right", "left"]
                ])
                delta = max(
                    [delta, abs(self.values[state] - newValues[state])]
                )
            iteration += 1
            self.values = newValues

            if visualize:
                print(f"{iteration = }\n{self}")
                # break

    def determinePolicy(self, treshold, discount, visualize):
        self.valueIteration(treshold, discount, visualize)

        for state in self.maze.states.flatten():
            bestReward = float("-inf")
            for direction in ["up", "down", "right", "left"]:
                newState = self.maze.states[self.maze.step(state.position, direction)]
                newReward = newState.reward + discount * self.values[newState]
                if newReward > bestReward:
                    bestReward = newReward
                    self.actions[state] = direction

    def select_action(self, state):
        return self.actions[state]

    def __str__(self) -> str:
        policy = super().__str__() + "\n\n"
        for x in range(len(self.maze.rewards)):
            for y in range(len(self.maze.rewards[0])):
                toprint = str(self.values[self.maze.states[(x,y)]]) + " " + self.select_action(self.maze.states[(x,y)])[:2]
                if (x, y) in self.maze.terminalpositions:
                    policy += colored(toprint, "green")
                else:
                    policy += toprint
                policy += ",\t"

            policy += "\n"
        return policy.rstrip()