# -*- coding: utf-8 -*-
"""

Created on 2019-02-22 16:05
@author Xiong Raorao

"""
from dataset import load_feature
from sklearn.cluster import *
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
    unsupervised evaluate score:  {'SC': 0.40285973649881957, 'CH': 6.693587946673831, 'DBI': 0.6624322266677443}

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

    # evaluate

    score = cluster_eval(gt_label, dbscan.labels_)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, dbscan.labels_)
    print('unsupervised evaluate score: ', unsupervise_score)
    return img_path, dbscan.labels_


def split_list(file_lists, num):
    '''
    split list to sublist
    '''
    interval = len(file_lists) // num
    mod = len(file_lists) % num
    ret = []
    for i in range(num):
        tmp = file_lists[i * interval:(i + 1) * interval]
        ret.append(tmp)
    return ret


def lfw2():
    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label, img_path = load_feature(path)

    # 分成5份依次聚类
    xx = split_list(X, 5)
    for i in range(5):
        db = DBSCAN(n_jobs=-1, eps=0.5, min_samples=1, metric='cosine')


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
    print('cost time: %d s' % round(time.time() - start))

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
    print('cost time: %d s' % round(time.time() - start))

    # evaluate

    score = cluster_eval(gt_label, birch.labels_)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, birch.labels_)
    print('unsupervised evaluate score: ', unsupervise_score)

    return img_path, birch.labels_


def lfw_agglomerative(linkage='complete'):
    '''
    # complete cosine
    gt n_cluster:  5740
    pre n_cluster:  5740
    cost time: 34 s

    evaluate score:  {'ARI': 0.8772158012260838, 'AMI': 0.9540866194718837, 'HCV': (0.9970073581889802, 0.9901151744759075, 0.993549313824344), 'FMI': 0.8836290092881737}
    unsupervised evaluate score:  {'SC': 0.3660289160550356, 'CH': 6.788387059880234, 'DBI': 0.6420190135471637}

    # average cosine
    gt n_cluster:  5740
    pre n_cluster:  5740
    cost time: 33 s

    evaluate score:  {'ARI': 0.9902014517124091, 'AMI': 0.985990567106264, 'HCV': (0.9975087289538368, 0.9969732346023019, 0.9972409098911819), 'FMI': 0.990228621069258}
    unsupervised evaluate score:  {'SC': 0.40472477697699766, 'CH': 6.732017848985664, 'DBI': 0.6379048022339981}

    # single cosine

    gt n_cluster:  5740
    pre n_cluster:  5740
    cost time: 30 s

    evaluate score:  {'ARI': 0.9882399651579442, 'AMI': 0.9857927097057834, 'HCV': (0.9969255699362045, 0.9971855683313122, 0.9970555521840598), 'FMI': 0.9882756231149297}
    unsupervised evaluate score:  {'SC': 0.40295206143306594, 'CH': 6.662192377284881, 'DBI': 0.7650319783317995}

    :return:
    '''

    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label, img_path = load_feature(path)
    agg = AgglomerativeClustering(n_clusters=5740, affinity='cosine', linkage=linkage)
    start = time.time()
    agg.fit(X)
    print('gt n_cluster: ', len(set(gt_label)))
    print('pre n_cluster: ', len(set(agg.labels_)))
    print('cost time: %d s' % round(time.time() - start))

    # evaluate

    score = cluster_eval(gt_label, agg.labels_)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, agg.labels_)
    print('unsupervised evaluate score: ', unsupervise_score)

    return img_path, agg.labels_


def lfw_AP():
    '''
    gt n_cluster:  5740
    pre n_cluster:  1723
    cost time: 275 s

    evaluate score:  {'ARI': 0.46572507910154803, 'AMI': 0.6476644589883233, 'HCV': (0.8942740529630575, 0.9747204563661763, 0.9327659430454032), 'FMI': 0.506312152565195}
    unsupervised evaluate score:  {'SC': 0.24097617806939176, 'CH': 7.922003068266033, 'DBI': 2.010803466033437}

    :return:
    '''

    path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    X, gt_label, img_path = load_feature(path)
    ap = AffinityPropagation(affinity='euclidean')
    start = time.time()
    ap.fit(X)
    print('gt n_cluster: ', len(set(gt_label)))
    print('pre n_cluster: ', len(set(ap.labels_)))
    print('cost time: %d s' % round(time.time() - start))

    # evaluate

    score = cluster_eval(gt_label, ap.labels_)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, ap.labels_)
    print('unsupervised evaluate score: ', unsupervise_score)

    return img_path, ap.labels_


def facescrub_db():
    '''

    dbscan train time: 508 s
    core cluster num:  92144
    gt n_cluster:  530
    pre n_cluster:  509

    evaluate score:  {'ARI': 0.7734228809345598, 'AMI': 0.9438636008104975, 'HCV': (0.9544715446091917, 0.9783882920765555, 0.9662819482731974), 'FMI': 0.7872490272398316}

    unsupervised evaluate score:  {'SC': 0.5033557155857413, 'CH': 292.44290237535967, 'DBI': 1.9704205586329755}
    :return:
    '''

    feature_path = 'C:\\Users\\xiongraorao\\Desktop\\facescrub_feature.txt'
    X, y, img_path = load_feature(feature_path)

    dbscan = DBSCAN(n_jobs=3, eps=0.5, min_samples=10, metric='cosine')
    start = time.time()
    dbscan.fit(X)

    print('dbscan train time: %d s' % round(time.time() - start))
    print('core cluster num: ', len(dbscan.core_sample_indices_))
    print('gt n_cluster: ', len(set(y)))
    print('pre n_cluster: ', len(set(dbscan.labels_)))

    # evaluate

    score = cluster_eval(y, dbscan.labels_)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, dbscan.labels_)
    print('unsupervised evaluate score: ', unsupervise_score)
    return img_path, dbscan.labels_


def visualize(files_path, pre_lables):
    # 写入数据库
    db = Mysql('192.168.1.11', 3306, 'root', '123456', 'face_reid')
    logger = Log('visualize', is_save=False)
    db.set_logger(logger)
    values = [[a, b.replace('/home/xrr/datasets/', '/data/'), c, d] for a, b, c, d in
              zip(pre_lables, files_path, ['2018-10-10 00:00:00'] * len(files_path), [1] * len(files_path))]
    sql = "truncate t_cluster2;insert into `t_cluster2` (`cluster_id`, `uri`, `timestamp`, `camera_id`) values {}".format(
        trans_sqlinsert(values))

    print(sql[:100])
    db.insert(sql)
    db.commit()
    print('visualize over')


def insert_ground_truth():
    '''
    写入真实样本的标签
    :param files_path:
    :param gt_labels:
    :return:
    '''
    # feature_path = 'C:\\Users\\xiongraorao\\Desktop\\facescrub_feature.txt'
    feature_path = 'C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt'
    _, y, img_path = load_feature(feature_path)
    return img_path, y


def evaluate_db():
    '''
    从数据库中得到pre_cluster
    :param gt:
    :return:
    '''
    feature_path = 'C:\\Users\\xiongraorao\\Desktop\\facescrub_feature.txt'
    _, y, img_path = load_feature(feature_path)
    pre_label = []
    db = Mysql('192.168.1.11', 3306, 'root', '123456', 'face_reid')
    logger = Log('visualize', is_save=False)
    db.set_logger(logger)
    from tqdm import tqdm
    for img in tqdm(img_path):
        file_name = img.split('/')[-1]
        sql = 'select cluster_id from t_cluster2 where uri like {}'.format("\"%" + file_name + "\"")
        label = db.select(sql)
        pre_label.append(label[0])

    # evaluate
    assert len(y) == len(pre_label), 'gt and predict label have different length'

    score = cluster_eval(y, pre_label)
    print('evaluate score: ', score)

img_path, pre_label = lfw()
# img_path, pre_label = lfw_kmeans()
# img_path, pre_label = lfw_birch()
# img_path, pre_label = lfw_agglomerative(linkage='single')
# img_path, pre_label = lfw_AP()

# img_path, pre_label = facescrub_db()

# img_path, pre_label = insert_ground_truth()

# visualize(img_path, pre_label)

# evaluate_db()