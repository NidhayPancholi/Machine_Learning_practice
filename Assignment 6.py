import pandas as pd
import numpy as np
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int,'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
train=pd.read_csv('kc_house_data_small_train.csv',dtype=dtype_dict)
test=pd.read_csv('kc_house_data_small_test.csv',dtype=dtype_dict)
valid=pd.read_csv('kc_house_data_validation.csv',dtype=dtype_dict)
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement',
                  'yr_built', 'yr_renovated','lat','long','sqft_living15','sqft_lot15']
def func(dataframe,features,output):
    train=pd.DataFrame()
    train['con']=[1 for x in range(len(dataframe))]
    for x in features:
        train[x]=dataframe[x]
    output_array=dataframe[output]
    return np.array(train),np.array(output_array)
def normalize(feature):
    return feature/np.linalg.norm(feature,axis=0),np.linalg.norm(feature,axis=0)
x_train,y_train=func(train,features,'price')
x_test,y_test=func(test,features,'price')
x_valid,y_valid=func(valid,features,'price')
X_train,norms=normalize(x_train)
print(norms)
X_valid=x_valid/norms
X_test=x_test/norms
query=X_test[0]
print(query)
print(X_train[9])
print(round(np.sqrt(sum((query-X_train[9])**2)),3))
def single_dist(query,feature):
    return np.sqrt(sum((query-feature)**2))
    #for x in feature:
     #   u.append(np.sqrt(sum((query-x)**2)))
    #return u
def dist(query,feature):
    return np.sqrt(np.sum((feature-query)**2,axis=1))
d=single_dist(query,X_train[9])
dist_10=dist(query,X_train[:10])
print(np.argsort(dist_10))
print(single_dist(query,X_train[100]))
query1=X_test[2]
dist_all=dist(query1,X_train)
print(np.argsort(dist_all)[0:5])
print('k=4 predicted value for third house in test set',round(np.average(y_train[np.argsort(dist_all)[0:4]])))
print('k=1 predicted value for third house in test set',y_train[382])
h=[]
for x in range(10):
    q=X_test[x]
    dist_x=dist(q,X_train)
    u=np.argsort(dist_x)[:10]
    sum=0
    for y in u:
        sum+=y_train[y]
    h.append(sum/10)
print(h)
print(h.index(min(h)))

print("___________________________________________")
h=[]

for k in range(1,16):
    j=[]
    for x in range(len(X_valid)):
        distance=dist(X_valid[x],X_train)
        u=np.argsort(distance)[:k]
        sum=0
        for i in u:
            sum+=y_train[u]
        j.append(((sum/k)-y_valid[x])**2)
    h.append(np.sum(j))
print(h)
print(h.index(min(h)))
print("____________________________________")
h=[]
for x in range(len(X_test)):
    dist1=dist(X_test[x],X_train)
    u=np.argsort(dist1)[0]
    h.append((y_train[u]-y_test[x])**2)
print(np.sum(h))








