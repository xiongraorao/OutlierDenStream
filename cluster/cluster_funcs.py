#-*- coding: utf-8 -*-
"""

Created on 2019-02-22 16:05
@author Xiong Raorao

"""
from dataset import load_feature
from sklearn.cluster import DBSCAN, KMeans, Birch
import time
import numpy as np
from util import cluster_eval, cluster_eval_withoutgt

def lfw():
    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label = load_feature(path)

    ## DBSCAN cluster
    dbscan = DBSCAN(n_jobs=-1, eps=0.5, min_samples=1, metric='cosine')
    start = time.time()

    dbscan.fit(X)
    print('dbscan train time: %d s' % round(time.time() - start))

    print('core cluster num: ', len(dbscan.core_sample_indices_))
    print('gt n_cluster: ', len(set(gt_label)))
    print('pre n_cluster: ', len(set(dbscan.labels_)))

    # evaluate

    score = cluster_eval(gt_label, dbscan.labels_)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, dbscan.labels_)
    print('unsupervised evaluate score: ', unsupervise_score)

lfw()



