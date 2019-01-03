from halite_logic import *
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output
import json
from keras.models import model_from_json, load_model
import numpy as np


def getStatus(state):
    # Evaluating game status
    status = 1
    bN, bi, bj = getLocation(state, 0, 1)
    sN, si, sj = getLocation(state, 3, 1)
    if (bi == si) & (bj == sj):
        status = 0
        print('return to shipyard')
    '''if iter > taille*taille:
        status = 0
        print('too many moves')
    if reward <= -1000:
        status = 0
        print('out of halite')'''
    if np.sum(state[0]) == 0 : # no boat
        status = 0
        print('boat collision')
    return status

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


def trainModel(model, epochs, gamma, epsilon, batchSize, buffer):
    '''
    epochs = 100 # nombre de parties
    gamma = 0.95
    epsilon = 1
    batchSize = 40
    buffer = 80'''
    replay = []
    #stores tuples of (S, A, R, S')
    h = 0
    model_name = 'model-grid-{}-epochs-{}-batchSize-{}.h5'.format(taille, epochs, batchSize)
    for i in range(epochs):
        state = initGrid(taille)
        qval = model.predict(state.reshape(1,taille*taille*4), batch_size=1)
        # 0-> eps to take random move, eps->1 to take the best Q move
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,5)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))

        # Take action, observe new state S'
        new_state, reward = makeMove2(state, action)

        # Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0 # reinitialize overwriting

            # feed replay memory
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1, 4*taille*taille), batch_size=1)
                newQ     = model.predict(new_state.reshape(1, 4*taille*taille), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,5))
                y[:] = old_qval[:]
                status = getStatus(new_state)
                if status == 1: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                    clear_output(wait=True)
                y[0][action] = update
                X_train.append(old_state.reshape(4*taille*taille,))
                y_train.append(y.reshape(5,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            print("--------------Game #: {}-------------------".format(i))
            # Train through batch
            model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=1)
            state = new_state
        status = getStatus(state)
        if reward != -1: #if reached terminal state, update game status
            status = 0
        clear_output(wait=True)

    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1/epochs)

    print('saving model...')
    model.save(model_name)
    print('model saved : ', model_name)
    return model
'''
# testing above functions
taille = 5
epochs = 100 # nombre de parties
gamma = 0.95
epsilon = 1
batchSize = 40
buffer = 80
model = initModel(taille)
model = trainModel(model, epochs, gamma, epsilon, batchSize, buffer)
'''
