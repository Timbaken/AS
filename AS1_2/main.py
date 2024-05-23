from maze import Maze
from agent import Agent
from policy import Policy
from valueiterationpolicy import Valueiterationpolicy

import numpy as np

rewards = np.array([[-1, -1, -1, 40],
                        [-1, -1, -10, -10],
                        [-1, -1, -1, -1],
                        [10, -2, -1, -1]])
terminalpositions = [(3, 0), (0, 3)]
startPosition = (3, 2)

maze = Maze(rewards, terminalpositions, startPosition)

policy = Policy()
treshold = 0.01
discount = 1
visualize = True
probability = 0.7
policy2 = Valueiterationpolicy(maze, treshold, discount, visualize, probability)

print(policy2)

agent = Agent(maze, policy2, probability)

print(agent, "\n")

for _ in range(100):
    agent.act()
    
    if agent.maze.states[agent.currentposition].terminal:
        break

print(f"score na het bereiken van terminal state: {agent.score}")
