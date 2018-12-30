from game_logic import *
import os
from IPython.display import clear_output
import numpy as np
import random
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(64,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?

model.add(Dense(150, kernel_initializer='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(4, kernel_initializer='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()


model.compile(loss='mean_squared_error', optimizer=rms)
epochs = 300
gamma = 0.95
epsilon = 1
batchSize = 40
buffer = 80
replay = []
#stores tuples of (S, A, R, S')
h = 0
for i in range(epochs):

    state = initGridPlayer() #using the harder state initialization function
    status = 1
    #while game still in progress
    while(status == 1):
        #We are in state S
        #Let's run our Q function on S to get Q values for all possible actions
        qval = model.predict(state.reshape(1,64), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else: #choose best action from Q(s,a) values
            action = (np.argmax(qval))
        #Take action, observe new state S'
        new_state = makeMove(state, action)
        #Observe reward
        reward = getReward(new_state)

        #Experience replay storage
        if (len(replay) < buffer): #if buffer not filled, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old values
            if (h < (buffer-1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            y_train = []
            for memory in minibatch:
                #Get max_Q(S',a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1,64), batch_size=1)
                newQ = model.predict(new_state.reshape(1,64), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,4))
                y[:] = old_qval[:]
                if reward == -1: #non-terminal state
                    update = (reward + (gamma * maxQ))
                else: #terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state.reshape(64,))
                y_train.append(y.reshape(4,))

            X_train = np.array(X_train)
            y_train = np.array(y_train)
            print("Game #: %s" % (i,))
            model.fit(X_train, y_train, batch_size=batchSize, epochs=1, verbose=1)
            state = new_state
        if reward != -1: #if reached terminal state, update game status
            status = 0
        clear_output(wait=True)
    if epsilon > 0.1: #decrement epsilon over time
        epsilon -= (1/epochs)