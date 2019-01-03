from halite_batch_training import *
import os
import glob

def last_file():
    path = os.getcwd()
    extension = 'h5'
    os.chdir(path)
    result = [i for i in glob.glob('*.{}'.format(extension))]
    print(result)

    file_dates = []
    for file in range(len(result)):
        file_dates.append(os.path.getmtime(result[file]))

    max_index = file_dates.index(max(file_dates))
    model_file = result[max_index]
    print('last modified file is {}'.format(model_file))
    if len(result) == 0:
        return None
    else:
        return model_file

file  = last_file()

if file is None:
    print('no file found, initializing new model')

    taille = 5
    epochs = 10 # nombre de parties
    gamma = 0.95
    epsilon = 1
    batchSize = 40
    buffer = 80
    model = initModel(taille)
    model = trainModel(model, epochs, gamma, epsilon, batchSize, buffer)
else:
    print('loading model from file')
    #from keras.models import load_model
    model = load_model(file)

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
    '''if iter > taille*taille:
        status = 0
        print('too many moves')'''
    if reward <= 0:
        status = 0
        print('out of halite')
    if np.sum(state[0]) == 0 : # no boat
        status = 0

print('------- END TESTING -------')


'''

def testModel(model, init=0):
    i = 0
    if init == 0:
        state = initGrid()
    elif init == 1:
        state = initGridPlayer()
    elif init == 2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    # while game still in progress
    while (status == 1):
        qval = model.predict(state.reshape(1, 64), batch_size=1)
        action = (np.argmax(qval))  # take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1  # If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
'''



