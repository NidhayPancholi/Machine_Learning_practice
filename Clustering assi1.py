import pandas as pd
import matplotlib.pyplot as plt          # plotting
import numpy as np   # dense matrices
from scipy.sparse import csr_matrix
from collections import Counter
df=pd.read_csv("people_wiki.csv")
def load_sparse_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    return csr_matrix((data, indices, indptr), shape)

word_count=load_sparse_csr('people_wiki_word_count.npz')

index=pd.read_json("people_wiki_map_index_to_word.json",orient='index')
from sklearn.neighbors import NearestNeighbors

model = NearestNeighbors(metric='euclidean', algorithm='brute')
model.fit(word_count)
distances, indices = model.kneighbors(word_count[35817], n_neighbors=10)
new_df=df.iloc[indices[0]]
print(new_df['name'])
word=[]
for x in df['text']:
    text=x.split(" ")
    word.append(Counter(text))
df['word_count']=word
def top_words(name):
    row = df[df['name'] == name]
    word_count_table = row[['word_count']].stack()
    #data={"words":word_count_table[0].keys,'counts':word_count_table[0].values()}
    data=dict(word_count_table[0])
    word_count_table=pd.DataFrame(data.items(),columns=['word','count'])
    return word_count_table.sort_values('count',ascending=False)

obama_words = top_words('Barack Obama')
barrio_words = top_words('Francisco Barrio')
combined_words=pd.merge(obama_words,barrio_words,on='word')
print(combined_words)
w={'the','in','and','of','to'}
def appear(w,col_name):
    u=0
    for x in df[col_name]:
        t=w.intersection(set(x.keys()))
        if len(t)==5:
            u+=1
    print(u)
bush_words=top_words('George W. Bush')
biden_words=top_words('Joe Biden')
t_combined=pd.merge(obama_words,bush_words,on='word',how='outer')
t_combined=pd.merge(t_combined,biden_words,on='word',how='outer')
t_combined.replace(np.NaN,0,inplace=True)
print(t_combined)
t_combined['bush_dist']=(t_combined['count_x']-t_combined['count_y'])**2
print(np.sqrt(np.sum(t_combined['bush_dist'])))
t_combined['biden_dist']=(t_combined['count_x']-t_combined['count'])**2
print(np.sqrt(np.sum(t_combined['biden_dist'])))
t_combined['biden_bush']=(t_combined['count_y']-t_combined['count'])**2
print(np.sqrt(np.sum(t_combined['biden_bush'])))

obama_bush=pd.merge(obama_words,bush_words,on='word')
print('_______________________________________')
tf_idf=load_sparse_csr('people_wiki_tf_idf.npz')
model_tf_idf = NearestNeighbors(metric='euclidean', algorithm='brute')
model_tf_idf.fit(tf_idf)
dist,ind=model_tf_idf.kneighbors(tf_idf[35817],n_neighbors=10)
new_df=df.iloc[ind[0]]
print(new_df['name'])


def unpack_dict(matrix,map_index_to_word):
    map_index_to_word=map_index_to_word.sort_values(0)
    table =map_index_to_word.index
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    num_doc = matrix.shape[0]
    return [{k: v for k, v in zip([table[word_id] for word_id in indices[indptr[i]:indptr[i + 1]]],
                                  data[indptr[i]:indptr[i + 1]].tolist())} \
            for i in range(num_doc)]
df['tf_idf'] = unpack_dict(tf_idf,index)
def top_words_tf_idf(name):
    row = df[df['name'] == name]
    word_count_table = row[['tf_idf']].stack()
    data=dict(word_count_table[0])
    word_count_table=pd.DataFrame(data.items(),columns=['word','tf_idf'])
    return word_count_table.sort_values('tf_idf',ascending=False)
obama_tf_idf=top_words_tf_idf('Barack Obama')
schiliro_tf_idf=top_words_tf_idf('Phil Schiliro')
obama_schiliro=pd.merge(obama_tf_idf,schiliro_tf_idf,on='word')
print(obama_schiliro)
appear({'obama','law','democratic','senate','presidential'},'tf_idf')
biden_tf_idf=top_words_tf_idf('Joe Biden')
obama_biden=pd.merge(obama_tf_idf,biden_tf_idf,on='word',how='outer')
obama_biden.replace(np.NaN,0)
print(obama_biden)
obama_biden['dist']=(obama_biden['tf_idf_x']-obama_biden['tf_idf_y'])**2
print(round(np.sqrt(np.sum(obama_biden['dist'])),3))
obama_schiliro['dist']=(obama_schiliro['tf_idf_x']-obama_schiliro['tf_idf_y'])**2
print(round(np.sqrt(np.sum(obama_schiliro['dist'])),3))
print(np.linalg.norm(obama_biden['tf_idf_x']-obama_biden['tf_idf_y']))