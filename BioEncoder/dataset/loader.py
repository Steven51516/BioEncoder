import numpy as np
import os
import wget
import pandas as pd
import json
import requests
from zipfile import ZipFile


def load_helper(path):
    result = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip()
            elements = line.split()
            result.append(elements)
            if(len(elements[3])>100):
                print(elements[0])
                print(elements[1])
                print(elements[2])
    data_transposed = list(map(list, zip(*result)))
    return data_transposed


# the data file should be in format drug, target, affinity
def load_DTI(path, id = False):
    data = load_helper(path)
    drugs = data[1]
    targets = data[2]
    affinity = data[3]
    ids = data[0]
    affinity = [float(i) for i in affinity]
    # affinity = convert_y_unit(np.array(affinity), 'nM', 'p')
    if not id:
        return np.array(drugs), np.array(targets), np.array(affinity)
    return np.array(ids), np.array(drugs), np.array(targets), np.array(affinity)


def load_DTI_CSV(path, id=False):
    # Load the CSV into a pandas DataFrame, skipping the first row
    df = pd.read_csv(path, skiprows=1, header=None)

    # Assuming the columns are ordered as id, drug, target, affinity
    drugs = df[0].values
    targets = df[1].values
    affinity = df[2].values.astype(float)

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

