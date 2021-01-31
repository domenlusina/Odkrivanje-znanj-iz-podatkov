import numpy as np
from scipy.cluster.vq import whiten
from scipy.sparse import *
from sklearn.cluster import KMeans


# preberemo train.mtx in jo shranimo kot npz -> hitrejše branje
def read_sparse(file='train.mtx'):
    data = np.loadtxt(file, skiprows=3, delimiter=' ')
    coo = coo_matrix((data[:, 2], (data[:, 0] - 1, data[:, 1] - 1)))
    csr = coo.tocsc()
    save_npz('train.npz', csr)
    return csr


# branje npz datoteke
def read_npz(file='train.npz'):
    sparse_matrix = load_npz(file)
    d_matrix = sparse_matrix.todense()
    return d_matrix


# pridobimo gosto matriko s podatki
dense_matrix = read_npz()
# normaliziramo vrstice
dense_matrix = whiten(dense_matrix.transpose())
dense_matrix = dense_matrix.transpose()

# usvarimo matriko above_zero, ki določi ali je element v matriki nad 0
above_zero = np.greater(dense_matrix, 0)

# pogledamo število elementov nad 0 v vsakem stolpcu
sum_above_zero = []
for i in range(dense_matrix[0].size):
    sum_above_zero.append(np.sum(above_zero[:, i]))
# izberemo tiste z največ elementi nad 0
indicies = (np.array(sum_above_zero)).argsort()[-1000:][::-1]
indicies.sort()

# ustvarimo novo matriko na podlagi prej pridobljenih stolpcev (indicies)
new_matrix2 = dense_matrix[:, indicies]
# new_matrix2 = above_zero[:, indicies]
dense_matrix = 0
# ponovno normaliziramo stolpce
w_new_matrix = whiten(new_matrix2.transpose())

# opravimo kmeans klustering
kmean = KMeans(n_clusters=40).fit(w_new_matrix.transpose())

# shranimo rezultate
thefile = open('test6_3.txt', 'w')
for item in kmean.labels_:
    thefile.write("%s\n" % item)
