from maze import Maze
from agent import Agent
from policy import Policy

import numpy as np

rewards = np.array([[-1, -1, -1, 40],
                        [-1, -1, -10, -10],
                        [-1, -1, -1, -1],
                        [10, -2, -1, -1]])
terminalpositions = [(3, 0), (0, 3)]
startPosition = (3, 2)

maze = Maze(rewards, terminalpositions, startPosition)

policy = Policy()

agent = Agent(maze, policy)

print(agent, "\n")

for _ in range(100):
    agent.act()
    
    if agent.maze.states[agent.currentposition].terminal:
        break

print(agent.score)