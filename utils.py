
import os
import copy
import pickle
import random

from config import log


## Data Load and Save

def save_pkl(data, name):
    fp = open(name, 'wb')
    pickle.dump(data, fp)
    fp.close()

def load_pkl(name):
    fp = open(name, 'rb')
    return pickle.load(fp)

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as rf:
        for line in rf:
            line = line.strip()
            data.append(line)
    return data

def save_data(data_list, path):
    with open(path, 'w', encoding='utf-8') as wf:
        for i, x in enumerate(data_list):
            x = x.strip()
            wf.write(x)
            if i != len(data_list) - 1:
                wf.write('\n')

def load_data_listdir(path_trans): # dataset only for this projecte (specifc form)
    log('> Loading')
    data = []
    for filename in os.listdir(path_trans):
        full_path = path_trans + '/' + filename
        each_file = open(full_path, 'r', encoding='utf-8')
        for x in each_file:
            if '#' == list(x)[0]:
                continue
            data.append(x.strip())
    return data

def split_data(data_list, ratio):
    """ ratio: 0 ~ 1
    """
    random.seed(777)
    random.shuffle(data_list) # shuffle
    pivot = int(len(data_list) * ratio)
    ldata = data_list[:pivot]
    rdata = data_list[pivot:]
    return ldata, rdata