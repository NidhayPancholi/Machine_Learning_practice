import pandas as pd
df=pd.read_csv('pandas_tutorial_read.csv',delimiter=';',header=None)
df.columns=['Date','read','country','id','source','topic']
df1=df[df['country']=='country_2']
print('Which source had the max reads?\n',df.groupby('source').count())
print('Which topic from which source had max reads in country_2?\n',df1.groupby(['source','topic']).count())
print("_______________________________________________________________")
df=df.drop(['Date','id','read'],axis=1)
print(df.head(2))
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
#import adspy_shared_utilities
d1=dict(zip(df['country'].unique(),(x for x in range(len(df['country'].unique())))))
d2=dict(zip(df['topic'].unique(),(x for x in range(len(df['topic'].unique())))))
df['country']=df['country'].replace(d1)
df['topic']=df['topic'].replace(d2)
print(df.head(2))
import matplotlib.pyplot as plt
h=[]
u=[]
k=[x for x in range(1,20)]
X_train,X_test,y_train,y_test=train_test_split(df[['country','topic']],df['source'],random_state=0)
for x in k:
    clf=KNeighborsClassifier(n_neighbors=x)
    clf.fit(X_train,y_train)
    h.append(clf.score(X_test,y_test))
    u.append(clf.score(X_train,y_train))
plt.plot(k,h,color='b',label='test')
plt.plot(k,u,color='r',label='train')
print("MAX test set accuracy=",max(h))
plt.legend()
plt.show()
u=[]
h=[]
o=[p for p in range(2,10)]
for x in o:
    tree=DecisionTreeClassifier(max_depth=x)
    tree.fit(X_train,y_train)
    h.append(tree.score(X_test,y_test))
    u.append(tree.score(X_train,y_train))
print('MAX test set accuracyusing tree=',max(h))
plt.plot(o,h,color='b',label='test')
plt.plot(o,u,color='r',label='train')
plt.legend()
plt.show()