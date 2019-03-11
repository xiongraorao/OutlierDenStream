# -*- coding: utf-8 -*-
"""

Created on 2019-03-05 17:34
@author Xiong Raorao

增量式的DBSCAN聚类方法

"""
import json
from itertools import islice, cycle

from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from cluster.util import cluster_eval, cluster_eval_withoutgt


class InDBSCAN():
    def __init__(self, n_jobs=None, eps=0.5, min_samples=1, metric='cosine'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.dbscan = DBSCAN(n_jobs=n_jobs, eps=eps, min_samples=min_samples, metric=metric)
        self.X = []
        self.initial = False

    def init_fit(self, X):
        '''
        offline cluster
        :return:
        '''
        # one shot dbscan
        if not self.initial:
            self.dbscan.fit(X)
            self.X.extend(X)
            self.initial = True
            self.labels_ = self.dbscan.labels_.tolist()
            self.core_sample_indices_ = self.dbscan.core_sample_indices_
            self.n_labels = len(set(self.labels_))

    def get_result(self):
        '''
        get cluster result
        :return:
        '''
        return self.dbscan.labels_

    def partial_fit(self, X):
        '''
        online cluster
        :param X shape: (n_samples, n_features_dim)
        :return:
        '''

        if not self.initial:
            raise RuntimeError('init_fit should be call before')

        # partial dbscan
        core_samples = [self.X[i] for i in self.core_sample_indices_]
        core_labels = [self.labels_[i] for i in self.core_sample_indices_]
        distances_matrix = pairwise.cosine_similarity(X, core_samples)
        distances = np.max(distances_matrix, axis=1)
        distances_index = np.argmax(distances_matrix, axis=1)

        # 老的cluster能够直达的label
        merge_samples = [X[i] for i, x in enumerate(distances) if x >= self.eps]
        merge_labels = [core_labels[y] for x, y in zip(distances, distances_index) if x >= self.eps]
        self.X.extend(merge_samples)
        self.labels_.extend(merge_labels)

        # 新的cluster能够知道的label
        new_samples = [X[i] for i, x in enumerate(distances) if x < self.eps]
        new_labels = list(range(self.n_labels, self.n_labels + len(new_samples)))
        self.X.extend(new_samples)
        self.labels_.extend(new_labels)


class Center():

    @staticmethod
    def vector_to_angle(A):
        '''
        求取向量A和每个坐标轴的夹角
        :param self:
        :param A: n_samples * n_features
        :type array-like
        :return: n_samples * n_features, radian angle
        :type: numpy.array
        '''
        if isinstance(A, list):
            A = np.array(A)
            if len(A.shape) == 1:
                A = A.reshape(1, -1)
        r = np.linalg.norm(A, axis=1).reshape(-1, 1)
        r.repeat(A.shape[1], axis=1)
        return np.arccos(A / r)

    @staticmethod
    def angle_to_vector(angle, radius=1):
        '''
        求取向量夹角为angle的长度为radis的向量
        :param angle: n_samples * n_features
        :type array-like
        :param radius:
        :type float/int
        :return: n_samples * n_features
        :type numpy.array
        '''
        if isinstance(angle, list):
            angle = np.array(angle)
        return radius * np.cos(angle)

    @staticmethod
    def center(X, metric='cosine'):
        if metric == 'cosine':
            angles = Center.vector_to_angle(X)
            mean_angle = np.mean(angles, axis=0)
            center = Center.angle_to_vector(mean_angle)
        elif metric == 'euclidean':
            center = np.mean(X, axis=0)
        else:
            center = None
            raise ValueError('metric type error')
        return center


class Cluster():
    def __init__(self, X, metric='cosine'):
        '''
        初始化cluster
        :param X:
        :type nparray
        '''
        self.points = X
        self.n = len(self.points)
        self.metric = metric
        self.center = Center.center(X, metric=self.metric)

    def add(self, X):
        self.points.append(X)
        # update center
        self.center = Center.center(self.points, metric=self.metric)


class IDBSCAN():
    '''
    纯增量式DBSCAN算法
    '''

    def __init__(self, n_jobs=None, eps=0.5, min_samples=1, metric='cosine'):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs
        self.core_samples = None
        self.clustered = False
        self.is_partial = True

    def fit(self, X):
        dbscan = DBSCAN(n_jobs=self.n_jobs, eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        dbscan.fit(X)
        self.labels_ = dbscan.labels_
        self.is_partial = False

    def partial_fit(self, X):
        '''
        增量DBSCAN
        :param X:
        :return:
        '''
        if not self.is_partial:
            print('this instance is not partial dbscan')
            return

        dbscan = DBSCAN(n_jobs=self.n_jobs, eps=self.eps, min_samples=self.min_samples, metric=self.metric)
        dbscan.fit(X)
        partial_labels = dbscan.labels_

        # get cluster
        clusters = {}
        for i, label in enumerate(dbscan.labels_):
            if label in clusters:
                clusters[label].append(i)
            else:
                clusters[label] = [i]

        C = []
        C_labels = []
        for k, v in clusters.items():
            C.append(Cluster(X[v]))
            C_labels.append(k)

        # merge operation
        if self.clustered:
            print('merge to origin cluster')
            # merge
            partial_centers = np.asarray([x.center for x in C])
            distance = np.dot(self.centers, partial_centers.T)  # n_old * n_new
            max_index = np.argmax(distance, axis=0)  # n = n_new
            max_value = np.max(distance, axis=0)  # n = n_new
            max_index = max_index[max_value >= self.eps]  # 只保留符合要求的old 索引
            merge_indices = np.argwhere(max_value >= self.eps)
            merge_list = [(old, new[0]) for old, new in zip(max_index.tolist(), merge_indices.tolist())]  # old-new
            new_indices = np.argwhere(max_value < self.eps)  # 单独成类的

            for old, new in merge_list:
                # update center
                self.centers[old] = Center.center(np.append(self.centers[old].reshape(1, -1), partial_centers[new].reshape(1, -1), axis=0), metric=self.metric)
                old_label = self.labels_[old]
                new_label = C_labels[new]
                update_indices = np.argwhere(partial_labels == new_label).reshape(-1, ).tolist()
                # update new_cluster label
                partial_labels[update_indices] = old_label

            existed_label_len = np.max(self.labels_)
            for i, new in enumerate(new_indices):
                partial_labels[new] = i + existed_label_len

            self.labels_ = np.append(self.labels_, partial_labels)

        else:
            # run partical_fit at the first time
            print('first clustering')
            self.labels_ = dbscan.labels_
            self.centers = np.asarray([x.center for x in C])
            self.clustered = True

    def predict(self, X):
        '''
        类型预测
        :param X:
        :return:
        '''
        pass


def load_feature(path):
    '''
    加载lfw数据集
    :param path:
    :return:
    '''
    X = []
    lables = []
    img_paths = []
    for line in open(path):
        data = json.loads(line)
        X.append(data['feature'])
        lables.append(data['label'])
        img_paths.append(data['path'].strip())
    return X, lables, img_paths

def load_data(db):
    if db == 'blob':
        noisy_blob = datasets.make_blobs(n_samples=1000, random_state=8, centers=4)
        X = noisy_blob[0]
        y = noisy_blob[1]
        X = StandardScaler().fit_transform(X)
        return X,y
    elif db == 'lfw':
        #X, y, _ = load_feature('C:\\Users\\xiongraorao\\Desktop\\lfw_feature.txt')
        X, y, _ = load_feature('C:\\Users\\raorao\\Desktop\\lfw_feature.txt')
        return X,y

def idbscan_test():

    X,y = load_data('blob')

    plt.figure(figsize=(10, 10))

    # 传统的dbscan
    dbscan = DBSCAN(min_samples=10, eps=0.4, n_jobs=-1, metric='euclidean')
    y1 = dbscan.fit_predict(X)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(y) + 1))))

    # idbscan
    idbscan = IDBSCAN(min_samples=10, eps=0.4,n_jobs=-1, metric='euclidean')
    XX = []
    # for i in range(10):
    #     XX.append(X[i*100: (i+1)*100])
    # for x in XX:
    #     idbscan.partial_fit(x)
    idbscan.partial_fit(X[:500])
    idbscan.partial_fit(X[500:700])
    idbscan.partial_fit(X[700:1000])
    y2 = idbscan.labels_
    print('n cluster: %d'%(len(set(y2))))

    # evaluate
    score = cluster_eval(y, y1)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, y1)
    print('unsupervised evaluate score: ', unsupervise_score)

    score = cluster_eval(y, y2)
    print('evaluate score: ', score)

    unsupervise_score = cluster_eval_withoutgt(X, y2)
    print('unsupervised evaluate score: ', unsupervise_score)

    plt.subplot(2, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y])
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.title('ground truth')

    plt.subplot(2, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y1])
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.title('dbscan result')

    plt.subplot(2, 2, 3)
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y2])
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.title('idbscan result')

    plt.show()


idbscan_test()