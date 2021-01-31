import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors import KDTree


class Point:
    '''
    coordinate: coordinates of a point represented as a list
    label:      defines the labels for a given point in the following manner
                -2 ... undefined
                -1 ... outlier
                0<=... cluster id
    '''

    def __init__(self, coordinate=[], label=-2):
        self.label = label
        self.coordinate = coordinate


class DBSCAN():
    def __init__(self, min_samples=4, eps=0.1, metric='euclidean'):
        self.min_samples = min_samples
        self.eps = eps
        self.metric = metric

    def data2Point(self, X):
        '''
        :param X:  list of points
        :return: list of datatype Points, which consist of coordinate and label attribute
        '''
        res = []
        for point in X:
            res.append(Point(point))
        return res

    def fit_predict(self, X):
        # definition of  distance metric
        dist = DistanceMetric.get_metric(self.metric)
        # initialization of KDTree with corresponding metric
        tree = KDTree(X, metric=dist)
        cluster_counter = -1
        X = self.data2Point(X)
        for i, point in enumerate(X):
            if point.label == -2:
                neigh_ind, _ = tree.query_radius([point.coordinate], r=self.eps, return_distance=True,
                                                 sort_results=True)
                neigh_ind = neigh_ind[0]
                # we mark points with less neighbors min_samples as noise/outlier
                if neigh_ind.size < self.min_samples:
                    X[i].label = -1
                else:
                    cluster_counter += 1
                    X[i].label = cluster_counter
                    neigh_ind = neigh_ind[1:].tolist()
                    for j in neigh_ind:
                        # we mark neighbors that don't belong to any cluster as current cluster
                        if X[j].label < 0:
                            X[j].label = cluster_counter
                            q_neigh_ind, _ = tree.query_radius([X[j].coordinate], r=self.eps, return_distance=True,
                                                               sort_results=True)
                            q_neigh_ind = q_neigh_ind[0].tolist()
                            # we add current node as neighbor if density of neighbors is high enough
                            if len(q_neigh_ind) >= self.min_samples:
                                new_el = list(set(q_neigh_ind).difference(set(neigh_ind)))
                                neigh_ind.extend(new_el)

        return np.array([x.label for x in X])


# vrne urejen seznam razdalje do k-tega soseda
def k_dist(X, metric='euclidean', k=3):
    # definition of  distance metric
    dist = DistanceMetric.get_metric(metric)
    # initialization of KDTree with corresponding metric
    tree = KDTree(X, metric=dist)
    res = []
    for point in X:
        dist, _ = tree.query([point], k + 1)
        res.append(dist[0][-1])  # izklucimo primer sam iz seznama sosedov
    res.sort(reverse=True)

    return res


# metoda komolca na grafu poišče najvecji prelom
def elbow_method(dists):
    max_diff = 0
    max_ind = 0
    for i in range(len(dists) - 2):
        k1 = dists[i] - dists[i + 1]
        k2 = dists[i + 1] - dists[i + 2]
        if k1 - k2 > max_diff:
            max_diff = k1 - k2
            max_ind = i + 1
    return dists[max_ind]


colors = {-1: '#e6194b', 0: '#3cb44b', 1: '#ffe119', 2: '#0082c8', 3: '#f58231', 4: '#911eb4', 5: '#46f0f0',
          6: '#f032e6', 7: '#d2f53c', 8: '#fabebe'}

data = genfromtxt('data.txt', delimiter='\t')
data = data[:, :-1]

plt.subplot(121)
plt.title('Graf k-dist')
# izrisemo graf za k-dist
kdist0 = k_dist(data, k=2)
plt.plot(kdist0, label='k = 2')
kdist1 = k_dist(data, k=3)
plt.plot(kdist1, label='k = 3')
kdist2 = k_dist(data, k=4)
plt.plot(kdist2, label='k = 4')
plt.legend()
plt.xlabel('Vzorci (točke) urejene po razdalji')
plt.ylabel('Razdalja do k-tega soseda')

# na podlagi metode komolca dolocimo radij
elbow = elbow_method(kdist2)

# izris klustrov po metodi DBSCAN
plt.subplot(122)
db = DBSCAN(eps=elbow)
plt.title('Graf raztrosa')
plt.xlabel('eps=' + str(elbow)[0:5] + ', min_samples=' + str(db.min_samples))
labels = db.fit_predict(data)

# pridobimo vse unikatne klustre
unique_labels = list(set(labels))

plt.scatter(data[:, 0], data[:, 1], color=[colors[x] for x in labels])

# prikaz legende
handles = [mpatches.Patch(color=colors[x], label=x) for x in unique_labels]
plt.legend(handles=handles)
plt.show()
