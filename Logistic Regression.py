from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
iris=load_iris()
y=(iris['target']==2).astype(int)
for i in range(4):
    X = iris['data']
    print(iris['feature_names'][i])
    log=LogisticRegression()
    X=X[:,i]
    X_train,X_test,y_train,y_test=train_test_split(X.reshape(-1,1),y,random_state=0,test_size=0.1)
    log.fit(X_train,y_train)
    y_pred=log.predict(X_test.reshape(-1,1))
    print(y_test)
    print(y_pred)
    import matplotlib.pyplot as plt
    import numpy as np
    X_new=np.linspace(min(X),max(X),50)
    pro=log.predict_proba(X_new.reshape(-1,1))
    plt.plot(X_new,pro[:,1],color='b',label='virginica')
    plt.plot(X_new,pro[:,0],color='r',label='non_virginica')
    plt.xlabel(iris['feature_names'][i])
    plt.ylabel("Probability")
    plt.legend()
    plt.show()