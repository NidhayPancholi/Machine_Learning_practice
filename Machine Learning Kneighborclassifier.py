from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('insurance.csv')
print(df.head(5))
h=[]
d={'northeast':1, 'southwest':2, 'northwest':3, 'southeast':4}
for x in df['region']:
    h.append(d[x])
df['region']=h
h=[]
d1={'male':1,'female':2}
for x in df['sex']:
    h.append(d1[x])
df['sex']=h
print(df.head(5))
X=df[['age','bmi','children','charges','sex','region']]
y=df['smoker']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,train_size=0.8)
s=[]
train_score=[]
for x in range(1,102,2):
    knn = KNeighborsClassifier(n_neighbors=x)
    knn.fit(X_train,y_train)
    u=knn.score(X_test,y_test)
    w=knn.score(X_train,y_train)
    s.append(u)
    train_score.append(w)
k=[ x for x in range(1,102,2)]
print('Maximum model accuracy=',max(s))
plt.plot(k,s,color='red')
plt.plot(k,train_score,color='blue')
plt.show()
plt.scatter(X_train['charges'],y_train)
plt.show()



