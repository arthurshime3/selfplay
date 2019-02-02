import numpy as np
import random
from grid import Grid
from alice import Alice
from cm import CM

grid = Grid(25)
alice = Alice(25, grid)

max_traj_length = 200

i = 0
while i < max_traj_length and not grid.isAtGoal(alice.getState()):
    action = alice.selectAction()
    grid.updateAlice(alice, action)
    # print("alice doing her thing")
    # print("current state {}".format(alice.getState()))
    # print("current reward {}".format(alice.getNetReward()))
    i += 1
#
# if grid.isAtGoal(alice.getState()):
#     print("Alice found goal")
# else:
#     print("Alice did not find goal")
# print("Number of steps taken: {}".format(i))
# print("Net reward {}".format(alice.getNetReward()))
# print(alice.getTrajectory())

cm = CM(alice.getTrajectory())
cm.run()
