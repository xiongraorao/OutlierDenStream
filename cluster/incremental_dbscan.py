'''
增量式的DBSCAN算法
'''
import json
from sklearn.neighbors import NearestNeighbors
import numpy as np
from dataset import load_index
from dataset import seeds


def euclidenDistances(A, B):
    vecProd = np.dot(A, B.T)
    SqA = A ** 2
    sumSqA = np.sum(SqA, axis=1, keepdims=True)
    sumSqAEx = np.tile(sumSqA, (1, vecProd.shape[1]))

    SqB = B ** 2
    sumSqB = np.sum(SqB, axis=1, keepdims=True)
    sumSqBEx = np.tile(sumSqB.T, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def cos_dist(A, B):
    vecProd = np.dot(A, B.T)
    normA = np.linalg.norm(A, axis=1, keepdims=True)
    normA = np.tile(normA, (1, vecProd.shape[1]))
    normB = np.linalg.norm(B, axis=1, keepdims=True)
    normB = np.tile(normB.T, (vecProd.shape[0], 1))
    return vecProd/(normA*normB)


def findCircle(X):
    '''
    寻找连通域的个数以及具体的连通域
    :param X:
    :return:
    '''
    m, n = X.shape
    if m != n:
        print('illegal')
        return
    circleNum = 0
    cluster_circles = []
    hasVisited = [False] * m
    for i in range(m):
        if not hasVisited[i]:
            circles = []
            dfs(X, i, hasVisited, circles)
            circleNum += 1
            cluster_circles.append(circles)
    return circleNum, cluster_circles


def dfs(X, index, hasVisited, circles):
    hasVisited[index] = True
    circles.append(index)
    for k in range(X.shape[0]):
        if X[index, k] == 1 and not hasVisited[k]:
            dfs(X, k, hasVisited, circles)


def dbscan(X, eps, min_pts):
    '''
    dbscan algorithm
    :param X:
    :param eps:
    :param min_pts:
    :return:
    '''
    # 1. initialize core objects
    # neighbors_model = NearestNeighbors(n_neighbors=200, radius=eps, metric='cosine')
    neighbors_model = NearestNeighbors(n_neighbors=10, radius=eps)
    neighbors_model.fit(X)
    neighborhoods = neighbors_model.radius_neighbors(X, eps, return_distance=False)
    n_neighbors = np.array([len(neighbors) for neighbors in neighborhoods])
    core_samples = np.asarray(n_neighbors >= min_pts, dtype=np.uint8)

    # 2. merge cluster
    core_clusters = []
    for i, core_sample in enumerate(core_samples):
        if core_sample == 1:
            core_clusters.append(neighborhoods[i])

    X = np.array(X)
    last_circle = None
    iterations = 100
    while True:
        if iterations < 0:
            break
        print('迭代次数：', 100 - iterations)
        means = np.array(list(map(lambda x: np.mean(X[x], axis=0), core_clusters)))
        # norm = np.linalg.norm(means, axis=1, keepdims=True)
        # mean_matrix = np.dot(means, means.T) / (np.dot(norm, norm.T))
        mean_matrix = cos_dist(means, means)
        mean_matrix = np.asarray(mean_matrix > 0.8, dtype=np.int8)

        circle_num, circles = findCircle(mean_matrix)
        if last_circle is not None and last_circle == circle_num:
            break
        last_circle = circle_num
        # merge
        trash_set = set()
        for circle in circles:
            print('circle: ', circle)
            if len(circle) > 2:
                for i in range(1, len(circle)):
                    core_clusters[circle[0]] = np.append(core_clusters[0], core_clusters[circle[i]])
                    trash_set.add(circle[i])

        # delete merged cluster
        for trash in sorted(trash_set, reverse=True):
            core_clusters.pop(trash)

        iterations -= 1

    # 输出label
    index_lable = {}
    for i, cluster in enumerate(core_clusters):
        for sample in cluster:
            index_lable[sample] = i

    # 加上噪声点
    noise_samples = set(range(len(X))) - set(index_lable.keys())
    for noise_sample in noise_samples:
        index_lable[noise_sample] = -1

    labels = [index_lable[x] for x in sorted(index_lable.keys())]
    print('n_cluster: ', len(core_clusters))
    print('n_labels: ', len(labels))
    print('lables: ', labels)


#X = load_index('./index-2018-12-26.log')
X = seeds()

dbscan(X, 0.5, min_pts=5)
