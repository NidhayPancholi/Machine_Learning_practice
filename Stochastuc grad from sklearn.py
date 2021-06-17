from sklearn.preprocessing import PolynomialFeatures
from numpy.random import rand,randn
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor,LinearRegression
import matplotlib.pyplot as plt
X=rand(100,1)*6-3
poly=PolynomialFeatures(2)
X_train=poly.fit_transform(X)
y=0.75*(X**2)+0.8*X+randn(100,1)
reg=LinearRegression()
reg.fit(X_train,y)
X_new=np.linspace(-3,3,20).reshape(-1,1)
X_pred=poly.fit_transform(X_new)
y_pred=reg.predict(X_pred)

plt.scatter(X,y)
plt.plot(X_new,y_pred,color='r')
plt.show()