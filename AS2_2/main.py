from maze import Maze
from optimalPolicy import OptimalPolicy
from temporalDifferenceAgent import TemporalDifferenceAgent
from SARSAAgent import SARSAAgent
from QAgent import QAgent
from doubleQAgent import DoubleQAgent

import numpy as np

def td():
    rewards = np.array(
        [[-1, -1, -1, 40],
        [-1, -1, -10, -10],
        [-1, -1, -1, -1],
        [10, -2, -1, -1]]
    )
    terminalpositions = [(3, 0), (0, 3)]
    startPosition = (3, 2)

    maze = Maze(rewards, terminalpositions, startPosition)

    optimalPolicyActions = np.array(
        [["right", "right", "right", None  ],
        ["up",    "up",    "up",    "up"  ],
        ["up",    "up",    "left",  "left"],
        [None,    "up",    "up",    "up"  ]]
    )

    optimalPolicyDict = {
        maze.states[i][j]: optimalPolicyActions[i][j] 
        for i in range(len(maze.states)) for j in range(len(maze.states[0]))
    }

    agent = TemporalDifferenceAgent(maze, OptimalPolicy(optimalPolicyDict))
    agent.temporal_difference(iterations=1_000)

def SARSA():
    rewards = np.array(
        [[-1, -1, -1, 40],
        [-1, -1, -10, -10],
        [-1, -1, -1, -1],
        [10, -2, -1, -1]]
    )
    terminalpositions = [(3, 0), (0, 3)]
    startPosition = (3, 2)

    maze = Maze(rewards, terminalpositions, startPosition)

    agent = SARSAAgent(maze)
    agent.SARSA(iterations=1_000_000)

def q():
    rewards = np.array(
        [[-1, -1, -1, 40],
        [-1, -1, -10, -10],
        [-1, -1, -1, -1],
        [10, -2, -1, -1]]
    )
    terminalpositions = [(3, 0), (0, 3)]
    startPosition = (3, 2)

    maze = Maze(rewards, terminalpositions, startPosition)

    agent = QAgent(maze)
    agent.q_learning(iterations=1_000_000)

def doubleq(probability):
    rewards = np.array(
        [[-1, -1, -1, 40],
        [-1, -1, -10, -10],
        [-1, -1, -1, -1],
        [10, -2, -1, -1]]
    )
    terminalpositions = [(3, 0), (0, 3)]
    startPosition = (3, 2)

    maze = Maze(rewards, terminalpositions, startPosition, probability)

    agent = DoubleQAgent(maze)
    agent.q_learning(iterations=1_000_000)

if __name__ == "__main__":
    probability = 0.9
    # td()
    # SARSA()
    # q()
    doubleq(probability)
