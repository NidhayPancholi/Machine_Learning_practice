import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
import sys
import os
df=pd.read_csv('people_wiki.csv')
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    return csr_matrix( (data, indices, indptr), shape)


def get_initial_centroids(data, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = data.shape[0]
    rand_indices = np.random.randint(0, n, k)
    centroids = data[rand_indices, :].toarray()

    return centroids
tf_idf = load_sparse_csr('people_wiki_tf_idf.npz')
index=pd.read_json('people_wiki_map_index_to_word.json',orient='index')
tf_idf = normalize(tf_idf)
def cal_dist(centres,point):
    dist=pairwise_distances(centres,point)
    return dist
all_dist=cal_dist(tf_idf[:3,:],tf_idf)
def find_nearest_cluster(dist_matrix):
    clusters=np.argmin(dist_matrix,axis=0)
    return clusters
clusters=find_nearest_cluster(all_dist)
def assign_clusters(centres,point):
    return find_nearest_cluster(cal_dist(centres,point))

def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
        member_data_points =data[cluster_assignment==i]
        centroid=np.mean(member_data_points,axis=0)
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    return new_centroids.reshape(k,547979)
result = revise_centroids(tf_idf[0:100:10], 3, np.array([0, 1, 1, 0, 0, 2, 0, 2, 2, 1]))
print(result.reshape(3,547979).shape)
def compute_heterogeneity(data, k, centroids, cluster_assignment):
    heterogeneity = 0.0
    for i in range(k):
        member_data_points = data[cluster_assignment == i, :]
        if member_data_points.shape[0]>0:
        distances = pairwise_distances(member_data_points,centroids[i])
        squared_distances = distances ** 2
        heterogeneity += np.sum(squared_distances)

    return heterogeneity


def kmeans(data, k, initial_centroids, maxiter, record_heterogeneity=None, verbose=False):
    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    h=[]

    for itr in range(maxiter):
        cluster_assignment = assign_clusters(centroids,data)
        centroids = revise_centroids(data,k,cluster_assignment)
        if prev_cluster_assignment is not None and \
                (prev_cluster_assignment == cluster_assignment).all():
            break
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment != cluster_assignment)
            #if verbose:
                #print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))
        if record_heterogeneity is not None:
            score = compute_heterogeneity(data,k,centroids,cluster_assignment)
            record_heterogeneity.append(score)
        prev_cluster_assignment = cluster_assignment[:]
        h.append(max(np.bincount(prev_cluster_assignment)))

    return centroids, cluster_assignment,h

k = 3
heterogeneity = []
initial_centroids = get_initial_centroids(tf_idf, k, seed=0)
centroids, cluster_assignment = kmeans(tf_idf, k, initial_centroids, maxiter=400,
                                       record_heterogeneity=None, verbose=True)
def smart_initialize(data, k, seed=None):
    if seed is not None:
        np.random.seed(seed)
    centroids = np.zeros((k, data.shape[1]))
    idx = np.random.randint(data.shape[0])
    centroids[0] = data[idx, :].toarray()
    squared_distances = pairwise_distances(data, centroids[0:1], metric='euclidean').flatten() ** 2
    for i in range(1, k):
        idx = np.random.choice(data.shape[0], 1, p=squared_distances / sum(squared_distances))
        centroids[i] = data[idx, :].toarray()
        squared_distances = np.min(pairwise_distances(data, centroids[0:i + 1], metric='euclidean') ** 2, axis=1)

    return centroids

def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('# Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.tight_layout()
for x in [0, 20000, 40000, 60000, 80000, 100000, 120000]:
    print(x)
    initial_centroids = get_initial_centroids(tf_idf, 3, seed=x)
    centroids,cluster_assignment,max_bins= kmeans(tf_idf,3,initial_centroids,400,verbose=False)
    print(max(max_bins))
