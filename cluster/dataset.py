#-*- coding: utf-8 -*-
"""

Created on 2019-02-19 17:47
@author Xiong Raorao

"""
import pandas as pd
import json


def seeds():
    path = './seeds_dataset.txt'
    d = pd.read_csv(path, sep='\t', header=-1)
    X = d.values[:, :-1]
    return X

def load_index(path):
    vectors = []
    with open(path, 'r') as f:
        for line in f:
            vector = json.loads(line.strip())['vector']
            vectors.append(vector)
    return vectors