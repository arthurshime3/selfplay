# Basic grid environment
# Consists of a start position, goal position, and key position
# Alice wants to find the key and then reach the goal in the least amount of steps possible

import numpy as np
import random


class Grid:
    def __init__(self, dim):
        self.grid = np.zeros([dim, dim])
        self.actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        self.state_actions = []
        for i in range(len(self.grid)):
            self.state_actions.append([])
            for j in range(len(self.grid[0])):
                valid_actions = []
                if i > 0:
                    valid_actions.append('UP')
                if i < len(self.grid) - 1:
                    valid_actions.append('DOWN')
                if j > 0:
                    valid_actions.append('LEFT')
                if j < len(self.grid[0]) - 1:
                    valid_actions.append('RIGHT')
                self.state_actions[i].append(valid_actions)

        self.goal_pos = [0, random.randint(0, len(self.grid) - 1)]
        self.start_pos = [len(self.grid) - 1, random.randint(0, len(self.grid) - 1)]
        self.key_pos = [random.randint((int)(len(self.grid) / 4), (int)(len(self.grid) / 4 * 3)), random.randint(0, (int)(len(self.grid) - 1))]
        self.grid[self.goal_pos[0]][self.goal_pos[1]] = 10
        self.key_found = False

    def getLegalActions(self, state):
        return self.state_actions[state[0]][state[1]]

    def updateAlice(self, alice, action):
        old_state = alice.getState()
        new_state = []
        if action == 'UP':
            new_state = [old_state[0] - 1, old_state[1]]
        elif action == 'DOWN':
            new_state = [old_state[0] + 1, old_state[1]]
        elif action == 'LEFT':
            new_state = [old_state[0], old_state[1] - 1]
        else:
            new_state = [old_state[0], old_state[1] + 1]
        alice.updateState(new_state)
        reward_update = -1
        if self.isAtGoal(alice.getState()):
            reward_update = 100
        if self.isAtKeyPos(alice.getState()):
            self.setKeyFound(True)
        alice.updateReward(reward_update)

    def getStartState(self):
        return self.start_pos

    def getGoalState(self):
        return self.goal_pos

    def getKeyPos(self):
        return self.key_pos

    def isAtGoal(self, state):
        return self.goal_pos == state and self.key_found

    def isAtKeyPos(self, state):
        return self.key_pos == state

    def setKeyFound(self, found):
        self.key_found = found
