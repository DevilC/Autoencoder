import cPickle as pickle
import numpy as np
import theano

def get_X_Y(data):
    X = np.asarray(data[0],dtype=theano.config.floatX)
    Y = np.asarray(data[1],dtype=theano.config.floatX)
    return X,Y

try:
    file = open(r"/home/cc/Documents/Theano_Learning/data/mnist.pkl")
except IOError:
    print "file don't exit!"

if file.closed is False:
    train_set, valid_set, test_set = pickle.load(file)
    train_setX, train_setY = get_X_Y(train_set)
    valid_setX, valid_setY = get_X_Y(valid_set)
    test_setX, test_setY = get_X_Y(test_set)

print "data loading!"


