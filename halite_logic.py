import numpy as np
import matplotlib.pyplot as plt
import time
'''
Spawn	Cost: 1000 halite
Convert into a drop-off	Cost: 4000 halite deducted from player’s stored halite.
The converted ship’s halite cargo and the halite in the sea under the new dropoff
is credited to the player, potentially reducing the cost.
Move: North, South, East, West	Cost: 10% of halite available at turn origin cell 
is deducted from ship’s current halite.
When a ship moves over a friendly shipyard or dropoff, it deposits its halite cargo.
Move: Stay still	Gain: 25% of halite available in cell, rounded up to the 
nearest whole number. Ship remains at its origin.
Ships can carry up to 1000 halite. 
'''

def initGrid(taille):
    np.random.seed(3)
    state = np.zeros((4, taille, taille), dtype=np.int)
    print(state.shape)
    # place boat
    state[0, int(taille/2), int(taille/2)] = 1

    # place wall, destroy boat
    state[1, taille - int(taille/2), taille- int(taille/2)] = 1
    state[1, taille - int(taille/2)-1, taille- int(taille/2)] = 1
    state[1, taille - int(taille/2)-2, taille- int(taille/2)] = 1
    # place halite map
    state[2,:,:] = np.random.randint(10, 500, size = (taille,taille))

    #state[3, 0, 0] = 1

    # place shipyard
    places = np.copy(state)
    places[2, : , :] = 0
    places = places.sum(axis = 0)
    [i, j] = np.random.randint(0,taille-1,2)
    while places[i,j] != 0 :
        [i, j] = np.random.randint(0,1,2)
    state[3, i, j] = 1

    '''
    # place boat
    state[0, 1] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place halite
    state[1, 1] = np.array([0, 1, 0, 0])
    # place shipyard
    state[3, 3] = np.array([1, 0, 0, 0])
    '''
    #print(state)
    return state

def getLocation(state, level, value):
    #        boat, wall, halite, shipyard
    # level :0,    1   , 2,    , 3
    locs = np.where(state[level]==value)
    N = len(locs[0])
    i = []
    j = []
    for k in range(N):
        i.append(locs[0][k])
        j.append(locs[1][k])
    return N, i, j

'''
def getLoc(state, level):
    
    taille = state.shape[-1]
    for i in range(0, taille):
        for j in range(0, 4):
            if (state[level][i, j] == 1):
                return i, j
def findLoc(state, level, value):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[level][i, j] == value).all():
                return i, j
                
'''

def makeMove(state, action):
    #        boat, wall, halite, shipyard
    # level :0,    1   , 2,    , 3
    taille = state.shape[-1]

    bN, bi, bj = getLocation(state, 0, 1)
    wN, wi, wj = getLocation(state, 1, 1)
    sN, si, sj = getLocation(state, 3, 1)

    #state = np.zeros((4, 4, 4))
    actions = [[-1, 0], [1, 0], [0, 0], [0, -1], [0, 1]]
    new_loc = bi[0] + actions[action][0], bj[0] + actions[action][1]
    print(new_loc)
    # move only if not wall, and new pos does not exceed grid !
    inGrid = (np.array(new_loc) <= (taille-1, taille-1)).all() & (np.array(new_loc) >= (0, 0)).all()
    if inGrid :
        if state[1][new_loc] == 0 & state[3][new_loc] == 0: # not a wall nor shipyard
            # erase old location
            state[0][bi,bj] = 0
            # move boat
            state[0][new_loc] = 1
        #elif state[3][new_loc] == 1 : # reached the shipyard
        #    state[0][new_loc] = 0
        #    status = False


    return state


def reward(state, action):
    if action in [0, 1, 3, 4]: # moving
        # get boat location
        bN, bi, bj = getLocation(state, 0, 1)
        # look for halite below boat
        halite = state[2][bi,bj]

        return -round(0.1*halite[0])
    else: # styaing
        # get boat location
        bN, bi, bj = getLocation(state, 0, 1)
        # look for halite below boat
        halite = state[2][bi,bj]

        return round(0.25*halite[0])








def dispGrid(state):
    fig, ax = plt.subplots()


    '''ib, jb = findLoc(state, 0, 1)

    ax.text(jb, ib, 'B', va='center', ha='center', fontsize=20)'''
    blocs = np.where(state[0]==1)
    for i in range(len(blocs[0])):
        ax.text(blocs[1][i], blocs[0][i], 'B', va='center', ha='center', fontsize=20)
    wlocs = np.where(state[1]==1)
    for i in range(len(wlocs[0])):
        ax.text(wlocs[1][i], wlocs[0][i], 'W', va='center', ha='center', fontsize=20)

    slocs = np.where(state[3]==1)
    for i in range(len(slocs[0])):
        ax.text(slocs[1][i], slocs[0][i], 'S', va='center', ha='center', fontsize=20)

    ax.imshow(state[2]/500, cmap='Blues', vmin=0, vmax = 2)
    plt.show()
def dispGrid2(state, r):
    fig, ax = plt.subplots()


    '''ib, jb = findLoc(state, 0, 1)

    ax.text(jb, ib, 'B', va='center', ha='center', fontsize=20)'''
    blocs = np.where(state[0]==1)
    for i in range(len(blocs[0])):
        ax.text(blocs[1][i], blocs[0][i], 'B', va='center', ha='center', fontsize=20)
    wlocs = np.where(state[1]==1)
    for i in range(len(wlocs[0])):
        ax.text(wlocs[1][i], wlocs[0][i], 'W', va='center', ha='center', fontsize=20)

    slocs = np.where(state[3]==1)
    for i in range(len(slocs[0])):
        ax.text(slocs[1][i], slocs[0][i], 'S', va='center', ha='center', fontsize=20)

    ax.imshow(state[2]/500, cmap='Blues', vmin=0, vmax = 2)
    plt.title('reward = {}'.format(r))
    plt.show()

taille = 5
state = initGrid(taille)
r = 0
iter = 0
status = True
while status:
    iter = iter + 1
    action = np.random.randint(0,4)
    state = makeMove(state, action)
    r = np.max([r + reward(state, action), 1000])

    dispGrid2(state, r)
    # test if position is shipyard
    bN, bi, bj = getLocation(state, 0, 1)
    sN, si, sj = getLocation(state, 3, 1)
    if (bi == si) & (bj == sj):
        status = False
        print('return to shipyard')
    if iter > taille*taille:
        status = False
        print('too many moves')
    if r <= 0:
        status = False
        print('out of halite')
    if state[0][bi,bj] == 0 :
        status= False
        print('boat collapsed')

print('Final reward = {}'.format(r))


'''for i in range(30):
    print(i)
    time.sleep(0.1)
    action = np.random.randint(0,4)
    state = makeMove(state, action)
    r = r + reward(state, action)
    dispGrid2(state, r)
'''





'''
import matplotlib.pyplot as plt
import matplotlib.cm as cm

img = [] # some array of images
frames = [] # for storing the generated images
fig = plt.figure()
for i in xrange(6):
    frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
# ani.save('movie.mp4')
plt.show()
'''
