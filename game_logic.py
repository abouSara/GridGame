import numpy as np


def randPair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


def findLoc(state, obj):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j] == obj).all():
                return i, j


# Initialize stationary grid, all items are placed deterministically
def initGrid():
    state = np.zeros((4, 4, 4))
    # place player
    state[0, 1] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[3, 3] = np.array([1, 0, 0, 0])

    return state


# Initialize player in random location, but keep wall, goal and pit stationary
def initGridPlayer():
    state = np.zeros((4, 4, 4))
    # place player
    state[randPair(0, 4)] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[1, 2] = np.array([1, 0, 0, 0])

    a = findLoc(state, np.array([0, 0, 0, 1]))  # find grid position of player (agent)
    w = findLoc(state, np.array([0, 0, 1, 0]))  # find wall
    g = findLoc(state, np.array([1, 0, 0, 0]))  # find goal
    p = findLoc(state, np.array([0, 1, 0, 0]))  # find pit
    if (not a or not w or not g or not p):
        # print('Invalid grid. Rebuilding..')
        return initGridPlayer()

    return state


def initGridRand():
    state = np.zeros((4, 4, 4))
    # player
    state[0, 1] = np.array([0, 0, 0, 1])
    # wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[3, 3] = np.array([1, 0, 0, 0])

    a = findLoc(state, np.array([0, 0, 0, 1]))
    w = findLoc(state, np.array([0, 0, 1, 0]))
    g = findLoc(state, np.array([1, 0, 0, 0]))
    p = findLoc(state, np.array([0, 1, 0, 0]))

    if (not a or not w or not g or not p):
        # print('Invalid grid. Rebuilding..')
        return initGridRand()

    return state


def makeMove(state, action):
    player_loc = findLoc(state, np.array([0, 0, 0, 1]))
    wall = findLoc(state, np.array([0, 0, 1, 0]))
    goal = findLoc(state, np.array([1, 0, 0, 0]))
    pit = findLoc(state, np.array([0, 1, 0, 0]))

    state = np.zeros((4, 4, 4))
    actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]

    '''print(action)
    print(actions[action])
    print(player_loc)
    print(np.array(player_loc))'''
    new_loc = player_loc[0] + actions[action][0], player_loc[1] + actions[action][1]
    print(new_loc)
    # move only if not wall, and new pos does not exceed grid !
    if (new_loc != wall):
        if ((np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all()):
            state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0, 0, 0, 1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0, 0, 0, 1])
    # re-place pit
    state[pit][1] = 1
    # re-place wall
    state[wall][2] = 1
    # re-place goal
    state[goal][0] = 1

    return state


def getLoc(state, level):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j][level] == 1):
                return i, j


def getReward(state):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if (player_loc == pit):
        return -10
    elif (player_loc == goal):
        return 10
    else:
        return -1


def dispGrid(state):
    grid = np.zeros((4, 4), dtype='U')
    player_loc = findLoc(state, np.array([0, 0, 0, 1]))
    wall = findLoc(state, np.array([0, 0, 1, 0]))
    goal = findLoc(state, np.array([1, 0, 0, 0]))
    pit = findLoc(state, np.array([0, 1, 0, 0]))
    for i in range(0, 4):
        for j in range(0, 4):
            grid[i, j] = ' '

    if player_loc:
        grid[player_loc] = 'P'  # player
    if wall:
        grid[wall] = 'W'  # wall
    if goal:
        grid[goal] = '+'  # goal
    if pit:
        grid[pit] = '-'  # pit

    return grid

def testAlgo(init=0):
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break