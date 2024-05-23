from policy import Policy
from maze import Maze
import numpy as np
from termcolor import colored


class Valueiterationpolicy(Policy):
    def __init__(self, maze: Maze, treshold: float, discount: float, visualize: bool, probability: float) -> None:
        super().__init__()

        self.maze = maze
        self.actions = {state: "up" for state in self.maze.states.flatten()}
        self.values = {state: 0 for state in self.maze.states.flatten()}
        self.directions = ["up", "down", "right", "left"]

        self.determinePolicy(treshold, discount, visualize, probability)

    def valueIteration(self, treshold, discount, visualize, probability):
        delta = float("inf")
        iteration = 0
        
        while delta > treshold:
            delta = 0
            newValues = self.values.copy()
            for state in self.maze.states.flatten():
                if state.terminal:
                    newValues[state] = 0
                    continue

                newValues[state] = max(
                    [
                        probability * 
                        (
                            self.maze.states[
                                self.maze.step(state.position, direction)
                            ].reward + discount * 
                            self.values[
                                self.maze.states[
                                    self.maze.step(state.position, direction)
                                ]
                            ]
                        ) 
                        + sum(
                            [((1 - probability) / (len(self.directions) -1)) * 
                                (
                                    self.maze.states[
                                        self.maze.step(state.position, otherdirection)
                                    ].reward + discount * 
                                    self.values[
                                        self.maze.states[self.maze.step(state.position, otherdirection)]
                                    ]
                                ) for otherdirection in self.directions if otherdirection != direction
                            ] 
                        ) for direction in self.directions
                    ]
                )
                delta = max(
                    [delta, abs(self.values[state] - newValues[state])]
                )
            iteration += 1
            self.values = newValues

            if visualize:
                print(f"{iteration = }\n{self}")
                # break

    def determinePolicy(self, treshold, discount, visualize, probability):
        self.valueIteration(treshold, discount, visualize, probability)

        for state in self.maze.states.flatten():
            bestReward = float("-inf")
            for direction in self.directions:
                newState = self.maze.states[self.maze.step(state.position, direction)]
                newReward = probability * (newState.reward + discount * self.values[newState])
                newReward += sum(
                            [(1 - probability) * 
                                (
                                    self.maze.states[
                                        self.maze.step(state.position, otherdirection)
                                    ].reward + discount * 
                                    self.values[
                                        self.maze.states[self.maze.step(state.position, otherdirection)]
                                    ]
                                ) for otherdirection in self.directions if otherdirection != direction
                            ] 
                        )
                if newReward > bestReward:
                    bestReward = newReward
                    self.actions[state] = direction

    def select_action(self, state):
        return self.actions[state]

    def __str__(self) -> str:
        policy = super().__str__() + "\n"
        max_width = max(len(str(round(item, 2))) for item in self.values.values()) + 5
        for x in range(len(self.maze.rewards)):
            for y in range(len(self.maze.rewards[0])):
                toprint = str(round(self.values[self.maze.states[(x,y)]], 2)) + " " + self.select_action(self.maze.states[(x,y)])
                toprint = f"{toprint:<{max_width}}"
                if (x, y) in self.maze.terminalpositions:
                    policy += colored(toprint, "green")
                else:
                    policy += toprint

            policy += "\n"
        return policy.rstrip()