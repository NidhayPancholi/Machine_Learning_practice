import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
from copy import copy
import matplotlib.pyplot as plt
def cosine_distance(x, y):
    xy = x.dot(y.T)
    dist = xy/(norm(x)*norm(y))
    return 1-dist[0,0]
def norm(x):
    sum_sq=x.dot(x.T)
    norm=np.sqrt(sum_sq)
    return(norm)
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    return csr_matrix((data, indices, indptr), shape)
def generate_random_vectors(dim,num_vector):
    return np.random.randn(num_vector,dim)


corpus = load_sparse_csr('people_wiki_tf_idf.npz')
df=pd.read_csv("people_wiki.csv")
index=pd.read_json('people_wiki_map_index_to_word.json',orient='index')
random_vectors = generate_random_vectors(num_vector=16, dim=547979)


def train_lsh(data, num_vector=16, seed=None):
    dim = data.shape[1]
    if seed is not None:
        np.random.seed(seed)
    random_vectors = generate_random_vectors(num_vector, dim)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
    table = {}
    bin_index_bits = (data.dot(random_vectors) >= 0)
    bin_indices = bin_index_bits.dot(powers_of_two)
    for data_index, bin_index in enumerate(bin_indices):
        if bin_index not in table.keys():
            table[bin_index] =[data_index]
        elif bin_index in table.keys():
            table[bin_index].append(data_index)
    model = {'data': data,
             'bin_index_bits': bin_index_bits,
             'bin_indices': bin_indices,
             'table': table,
             'random_vectors': random_vectors,
             'num_vector': num_vector}
    return model
model1 = train_lsh(corpus, num_vector=16, seed=143)
print(model1)
table = model1['table']
print(df[df['name'] == 'Joe Biden'])
for x in table.keys():
    if 24478 in table[x]:
        print(x)
        break
print(model1['bin_index_bits'][35817])
print(model1['bin_index_bits'][24478])
from itertools import combinations
def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector - 1, -1, -1)
    candidate_set = copy(initial_candidates)
    candidate_set.update(table[query_bin_bits.dot(powers_of_two)])
    for different_bits in combinations(range(num_vector), search_radius):
        alternate_bits = copy(query_bin_bits)
        alternate_bits=alternate_bits.astype(int)
        print(alternate_bits)
        for i in different_bits:
            if alternate_bits[i]==0:
                alternate_bits[i] = 1
            elif alternate_bits[i]==1:
                alternate_bits[i]=0
        print(alternate_bits)
        nearby_bin = alternate_bits.dot(powers_of_two)
        if nearby_bin in table:
                candidate_set.update(table[nearby_bin])
    return candidate_set
obama_bin_index = model1['bin_index_bits'][35817]
candidate_set = search_nearby_bins(obama_bin_index, model1['table'], search_radius=1)
def query(vec, model, k, max_search_radius):
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)
    nearest_neighbors = pd.DataFrame(list(candidate_set))
    nearest_neighbors.rename(columns={0:'id'},inplace=True)
    candidates = data[np.array(list(candidate_set)), :]
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()
    return nearest_neighbors.nsmallest(k,'distance'),len(candidate_set)
print(query(corpus[35817,:], model1, k=10, max_search_radius=3))
result, num_candidates_considered = query(corpus[35817,:], model1, k=10, max_search_radius=2)
df['id']=df.index
result=pd.merge(result,df[['id','name']],on='id')
print(np.mean(result['distance']))