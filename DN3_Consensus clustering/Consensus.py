import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.sparse import lil_matrix, triu
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample


class ConsensusClustering:
    def __init__(self, max_K=5, H=10, cluster=KMeans(), resample=resample()):
        """   
         
        :param max_K: max number of clusters
        :param H: number of resampling interations
        :param cluster: clustering algorithm  (default. sklearn.cluster.KMeans)
        :param resample:  resampling algorithm (default. sklearn.utils.resample)
        """
        self.max_K = max_K
        self.cluster = cluster
        self.H = H
        self.resample = resample

    def run(self, data, N=-1):
        """
        
        :param data: data matrix where each row represents an item
        :param N: number of items
        :return: 
        """

        # data = shuffle(data)

        best_M = 0
        k_hat = 2
        old_delta = 0
        old_A = 0

        # if N is not given we assume each line represents an item
        if N < 0:
            N = data.shape[0]
        # we don't include 1 since it doesn't make much sense -> we want to split data into one group?
        for k in range(2, self.max_k + 1):
            M = lil_matrix((N, N))
            I = lil_matrix((N, N))
            for h in range(self.H):
                pert = resample(list(range(N)))
                D_h = data[pert, :]
                uniq_pert = list(set(pert))

                I_h = lil_matrix((N, N))
                I_h[np.ix_(uniq_pert, uniq_pert)] = 1
                I = I._add_sparse(I_h)

                # we set the number of clusters
                self.cluster.n_clusters = k
                res = self.cluster.fit(D_h)
                M_h = lil_matrix((N, N))
                labels = np.array(res.labels_)
                for ki in range(k):
                    indicies = np.where(labels == ki)[0]
                    M_h[np.ix_(indicies, indicies)] = 1
                M = M._add_sparse(M_h)

            # we look for a best k based on consensus distribution
            new_M = M._divide(I)
            new_A = self.A(new_M)
            if self.deltaK(new_A, old_A, k) > old_delta:
                k_hat = k
                best_M = new_M
            old_A = new_A

        part_of_data = data[0:N / 10, :]
        Z = linkage(part_of_data, 'ward')
        y = fcluster(Z, k_hat, criterion='maxclust')
        clf = RandomForestClassifier()
        clf.fit(Z, y)

        return clf.predict(data), best_M

    def CDF(self, C, c):
        """
        
        :param C: consensus matrix
        :param c: threshold for which we check if value in matrix C is bellow or not
        :returns: empirical cumulative distribution
        """

        N = C.shape[0]
        sum_C = triu(C <= c, k=1).sum()
        return sum_C / ((N - 1) * N / 2)

    def A(self, C):
        """
        
        :param C: consensus matrix
        :return: area under the CDF
        """
        N = C.shape[0]
        indicies = np.triu_indices(N, 1)  # upper-triangle indices
        tri_C = C[indicies]
        uniq_values = list(set(tri_C.data))
        S = 0
        for i in range(len(uniq_values) - 1):
            S += (uniq_values[i + 1] - uniq_values[i]) * self.CDF(C, uniq_values[i + 1])
        return S

    def deltaK(self, new_A, old_A, k):
        """
        
        :param old_A: area under the CDF 
        :param new_A: area under the CDF 
        :return: angle between points A1 and A2 on the graph
        """
        if k == 2:
            return old_A
        else:
            return (old_A - new_A) / new_A
