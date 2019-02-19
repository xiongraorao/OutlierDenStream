import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_index(path):
    vectors = []
    with open(path, 'r') as f:
        for line in f:
            vector = json.loads(line.strip())['vector']
            vectors.append(vector)
    return vectors

def dbscan(X):
    '''
    dbscan 聚类方法
    :return:
    '''
    # db = DBSCAN(eps=0.6, min_samples=10, metric='cosine').fit(X)
    X = np.array(X)
    S = 1 - np.dot(X, X.T)
    db = DBSCAN(eps=0.6, min_samples=5, metric='precomputed').fit(S)

    labels = db.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print('n_clusters: ', n_clusters)
    print('n_noise: ', n_noise_)
    print('labels: ', labels)
    print('db core sample index', len(db.core_sample_indices_))
    print('db algorithm', db.algorithm)
    return labels

X = load_index('./index-2018-12-26.log')

tsne = TSNE(n_components=3)


labels = dbscan(X)

Y = tsne.fit_transform(X)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c=plt.cm.Set1(labels / 100.))

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('t-sne')

print(plt.cm.Set1(labels / 10.))

plt.show()