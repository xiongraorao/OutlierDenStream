#-*- coding: utf-8 -*-
"""

Created on 2019-02-22 16:05
@author Xiong Raorao

"""
from dataset import load_feature
from sklearn.cluster import DBSCAN, KMeans, Birch, AgglomerativeClustering
import time
import numpy as np
from util import cluster_eval, cluster_eval_withoutgt, trans_sqlinsert
from mysql import Mysql
from logger import Log


def lfw():
    '''
    result:

    dbscan train time: 9 s
    core cluster num:  13211
    gt n_cluster:  5740
    pre n_cluster:  5768

    evaluate score:  {'ARI': 0.9884690711797678, 'AMI': 0.9851659125925958, 'HCV': (0.9973329806102722, 0.9967953906459855, 0.9970641131646472), 'FMI': 0.9885028490650399}
    unsupervised evaluate score:  0.40285973649881957

    :return:
    '''

    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label, img_path = load_feature(path)

    ## DBSCAN cluster
    dbscan = DBSCAN(n_jobs=-1, eps=0.5, min_samples=1, metric='cosine')
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
    # return img_path, dbscan.labels_

def lfw_kmeans():
    '''
    result:
    817s
    gt n_cluster:  5740
    pre n_cluster:  5740

    evaluate score:  {'ARI': 0.2946529011301506, 'AMI': 0.7480503460467403, 'HCV': (0.9914706899544379, 0.9471878835541047, 0.9688235331962477), 'FMI': 0.41122064384553325}
    unsupervised evaluate score:  0.2398293099762878
    :return:
    '''

    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label, img_path = load_feature(path)
    kmeans = KMeans(n_jobs=-1, n_clusters=5740)
    start = time.time()
    kmeans.fit(X)
    print('gt n_cluster: ', len(set(gt_label)))
    print('pre n_cluster: ', len(set(kmeans.labels_)))
    print('cost time: %d s'% round(time.time() - start))

    # # evaluate
    #
    # score = cluster_eval(gt_label, kmeans.labels_)
    # print('evaluate score: ', score)
    #
    # unsupervise_score = cluster_eval_withoutgt(X, kmeans.labels_)
    # print('unsupervised evaluate score: ', unsupervise_score)

    return img_path, kmeans.labels_

def lfw_birch():
    '''
    gt n_cluster:  5740
    pre n_cluster:  5740
    cost time: 16 s

    evaluate score:  {'ARI': 0.9088314702653802, 'AMI': 0.9457411154577685, 'HCV': (0.9950424072953659, 0.9883225758324755, 0.9916711078460287), 'FMI': 0.911949218026558}
    unsupervised evaluate score:  0.34134643599754744

    :return:
    '''
    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label, img_path = load_feature(path)
    birch = Birch(n_clusters=5740)
    start = time.time()
    birch.fit(X)
    print('gt n_cluster: ', len(set(gt_label)))
    print('pre n_cluster: ', len(set(birch.labels_)))
    print('cost time: %d s'% round(time.time() - start))

    # evaluate

    score = cluster_eval(gt_label, birch.labels_)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, birch.labels_)
    print('unsupervised evaluate score: ', unsupervise_score)

    return img_path, birch.labels_

def lfw_agglomerative():
    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label, img_path = load_feature(path)
    agg = AgglomerativeClustering(n_clusters=5740, affinity='cosine', )

def visualize(files_path, pre_lables):
    # 写入数据库
    db = Mysql('192.168.1.11', 3306, 'root', '123456', 'face_reid')
    logger = Log('visualize', is_save = False)
    db.set_logger(logger)
    values = [[a,b.replace('/home/xrr/datasets/', '/data/'),c,d] for a,b,c,d in zip(pre_lables,files_path, ['2018-10-10 00:00:00']* len(files_path), [1]* len(files_path))]
    print(values[:2])
    sql = "truncate t_cluster2;insert into `t_cluster2` (`cluster_id`, `uri`, `timestamp`, `camera_id`) values {}".format(trans_sqlinsert(values))

    print(sql[:100])
    db.insert(sql)
    db.commit()
    print('visualize over')

# img_paths, pre_label = lfw()
# img_path, pre_label = lfw_kmeans()
img_path, pre_label = lfw_birch()

visualize(img_path, pre_label.tolist())

