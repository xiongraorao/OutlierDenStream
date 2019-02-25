#-*- coding: utf-8 -*-
"""

Created on 2019-02-22 16:05
@author Xiong Raorao

"""
from dataset import load_feature
from sklearn.cluster import DBSCAN, KMeans, Birch
import time
import numpy as np
from util import cluster_eval, cluster_eval_withoutgt, trans_sqlinsert
from mysql import Mysql
from logger import Log


def lfw():
    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label, img_path = load_feature(path)

    ## DBSCAN cluster
    dbscan = DBSCAN(n_jobs=-1, eps=0.5, min_samples=1, metric='cosine')
    KMeans(n_jobs=-1, n_clusters=5740)
    start = time.time()

    dbscan.fit(X)
    print('dbscan train time: %d s' % round(time.time() - start))

    print('core cluster num: ', len(dbscan.core_sample_indices_))
    print('gt n_cluster: ', len(set(gt_label)))
    print('pre n_cluster: ', len(set(dbscan.labels_)))

    # # evaluate
    #
    # score = cluster_eval(gt_label, dbscan.labels_)
    # print('evaluate score: ', score)
    #
    # unsupervise_score = cluster_eval_withoutgt(X, dbscan.labels_)
    # print('unsupervised evaluate score: ', unsupervise_score)
    return img_path, dbscan.labels_


def visualize(files_path, pre_lables):
    # 写入数据库
    db = Mysql('192.168.1.11', 3306, 'root', '123456', 'face_reid')
    logger = Log('visualize', is_save = False)
    db.set_logger(logger)
    values = [[a,b.replace('/home/xrr/datasets/', '/data/'),c,d] for a,b,c,d in zip(pre_lables,files_path, ['2018-10-10 00:00:00']* len(files_path), [1]* len(files_path))]
    print(values[:2])
    sql = "insert into `t_cluster2` (`cluster_id`, `uri`, `timestamp`, `camera_id`) values {}".format(trans_sqlinsert(values))

    print(sql[:100])
    db.insert(sql)
    db.commit()
    print('visualize over')

img_paths, pre_label = lfw()

visualize(img_paths, pre_label.tolist())
