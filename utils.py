
import os
import pickle


def save_pkl(data, name):
    fp = open(name, 'wb')
    pickle.dump(data, fp)
    fp.close()

def load_pkl(name):
    fp = open(name, 'rb')
    return pickle.load(fp)




