import random

class Alice:
    def __init__(self, dim, grid):
        self.maxPathLength = dim * 2
        self.grid = grid
        self.state = grid.getStartState()
        self.total_reward = 0
        self.trajectory = []

    def selectAction(self):
        legal_actions = self.grid.getLegalActions(self.state)
        action = random.choice(legal_actions)
        self.trajectory.append((self.state, action))
        return action

    def updateState(self, state):
        self.state = state

    def updateReward(self, reward):
        self.total_reward += reward

    def getTrajectory(self):
        return self.trajectory

    def getState(self):
        return self.state

    def getNetReward(self):
        return self.total_reward
