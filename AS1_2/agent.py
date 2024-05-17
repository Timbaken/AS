from maze import Maze
from policy import Policy
from termcolor import colored

class Agent:
    def __init__(self, maze: Maze, policy: Policy) -> None:
        self.policy = policy
        self.maze = maze
        self.currentposition = maze.startPosition
        self.score = 0

    def act(self):
        action = self.policy.select_action(self.maze.states[self.currentposition[0]][self.currentposition[1]])
        self.currentposition = self.maze.step(self.currentposition, action)
        self.score += self.maze.rewards[self.currentposition[0]][self.currentposition[1]]
        print(action)
        print(self, "\n")

    def __str__(self) -> str:
        maze = ""
        for x in range(len(self.maze.rewards)):
            for y in range(len(self.maze.rewards[0])):
                if (x, y) == self.currentposition:
                    maze += colored(self.maze.rewards[x][y], "yellow")
                elif (x, y) in self.maze.terminalpositions:
                    maze += colored(self.maze.rewards[x][y], "green")
                elif self.maze.rewards[x][y] < -5:
                    maze += colored(self.maze.rewards[x][y], "red")
                else:
                    maze += str(self.maze.rewards[x][y])
                maze += ", "

            maze += "\n"
        return maze.rstrip()

