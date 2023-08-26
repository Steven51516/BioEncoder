import numpy as np
import os
import wget
import pandas as pd
import json


def load_helper(path):
    result = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            elements = line.split()
            result.append(elements)
    data_transposed = list(map(list, zip(*result)))
    return data_transposed


# the data file should be in format drug, target, affinity
def load_DTI(path):
    data = load_helper(path)
    drugs = data[0]
    targets = data[1]
    affinity = data[2]
    affinity = [float(i) for i in affinity]
    affinity = convert_y_unit(np.array(affinity), 'nM', 'p')
    return np.array(drugs), np.array(targets), np.array(affinity)


def convert_y_unit(y, from_, to_):
    array_flag = False
    if isinstance(y, (int, float)):
        y = np.array([y])
        array_flag = True
    y = y.astype(float)
    # basis as nM
    if from_ == 'nM':
        y = y
    elif from_ == 'p':
        y = 10**(-y) / 1e-9

    if to_ == 'p':
        zero_idxs = np.where(y == 0.)[0]
        y[zero_idxs] = 1e-10
        y = -np.log10(y*1e-9)
    elif to_ == 'nM':
        y = y

    if array_flag:
        return y[0]
    return y