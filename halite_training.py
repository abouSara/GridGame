from halite_logic import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import numpy as np

def initModel(taille):
    inputs = taille*taille*4
    model = Sequential()
    model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(inputs,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

    model.add(Dense(150, kernel_initializer='lecun_uniform'))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(5, kernel_initializer='lecun_uniform'))
    model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

    rms = RMSprop()
    model.compile(loss='mean_squared_error', optimizer=rms)

    return model

    #model.predict(state.reshape(1,64), batch_size=1)

taille = 5
state = initGrid(taille)
status = 1
model = initModel(taille)

epochs = 100
gamma = 0.9  # since it may take several moves to goal, making gamma high
epsilon = 1

print('-----TRAINING-----')
for i in range(epochs):
    #status = 1
    reward = 500
    # while game still in progress
    while (status == 1):

        # We are in state S
        # Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1, taille*taille*4), batch_size=1)
        print("qval = {}".format(qval))
        if (np.random.random() < epsilon):  # choose random action
            action = np.random.randint(0, 4)
        else:  # choose best action from Q(s,a) values
            action = (np.argmax(qval))
        # Take action, observe new state S'

        new_state, reward = makeMove2(state, action)

        # Get max_Q(S',a)
        newQ = model.predict(new_state.reshape(1, taille*taille*4), batch_size=1)
        maxQ = np.max(newQ)
        y = np.zeros((1, 5))
        y[:] = qval[:]


        bN, bi, bj = getLocation(state, 0, 1)
        sN, si, sj = getLocation(state, 3, 1)
        if (bi == si) & (bj == sj):
            status = 0
            print('return to shipyard')
        if iter > taille*taille:
            status = 0
            print('too many moves')
        if reward <= -1000:
            status = 0
            print('out of halite')
        if np.sum(state[0]) == 0 : # no boat
            status = 0



        if status ==1:  # non-terminal state
            update = (reward + (gamma * maxQ))
        else:  # terminal state
            update = reward
            clear_output(wait=True)

        y[0][action] = update  # target output

        model.fit(state.reshape(1, taille*taille*4), y, batch_size=1, epochs=1, verbose=1)
        state = new_state


    if epsilon > 0.1:
        epsilon -= (1 / epochs)


print('-----TESTING-----')

taille = 5
state = initGrid(taille)
status = 1
reward = 500
while status ==1:
    qval = model.predict(state.reshape(1,4*taille*taille), batch_size=1)
    action = np.argmax(qval)
    state, r = makeMove2(state, action)
    reward = np.min([reward+r, 1000])
    dispGrid2(state, r)
    bN, bi, bj = getLocation(state, 0, 1)
    sN, si, sj = getLocation(state, 3, 1)
    if (bi == si) & (bj == sj):
        status = 0
        print('return to shipyard')
    if iter > taille*taille:
        status = 0
        print('too many moves')
    if reward <= 0:
        status = 0
        print('out of halite')
    if np.sum(state[0]) == 0 : # no boat
        status = 0

print('------- END TESTING -------')


'''
## popo
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
'''