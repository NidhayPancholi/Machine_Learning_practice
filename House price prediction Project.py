import pandas as pd
import numpy as np
import  matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,Ridge,LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,SCORERS,mean_squared_error
from sklearn.svm import  SVR
df=pd.read_csv('kc_house_data.csv')
print(SCORERS.keys())
df=df[df['price']<1.5e6]
df=df[df['sqft_living']<=4000]
df['sqft*2']=df['sqft_living']**2
dummy=pd.get_dummies(df['zipcode'])
df=pd.concat([df.drop('zipcode',axis=1),dummy],axis=1)
df['bed_bath_floors']=round(df['bathrooms'])+5*df['bedrooms']+df['floors']   # gives 5 times more priority to bedroom compared to bathrooms
df['grade_cond']=(df['grade']*df['condition']/10)
df.drop(['id','date','sqft_lot15','sqft_living15','sqft*2','bed_bath_floors','grade_cond'],axis=1,inplace=True)
from sklearn.model_selection import train_test_split,cross_val_score
X_train,x_test,y_train,Y_test=train_test_split(df.drop('price',axis=1),df['price'],test_size=0.2,random_state=0)
X_valid,X_test,y_valid,y_test=train_test_split(x_test,Y_test,test_size=0.25,random_state=0)
print(X_train[:5])
from numpy.linalg import norm
print(X_train.columns)
from sklearn.ensemble import RandomForestRegressor
forest_df=df
f_X_train,f_X_test,f_y_train,f_y_test=train_test_split(forest_df.drop("price",axis=1),forest_df['price'],random_state=0)
#for x in range(10,20):
 #   forest=RandomForestRegressor(max_depth=x)
  #  forest.fit(X_train,y_train)
   # forest_pred=forest.predict(X_valid)
    #print(x,' =',mean_absolute_error(y_valid,forest_pred))
forest_final=RandomForestRegressor(max_depth=20)
forest_final.fit(X_train,y_train)
forest_test_pred=forest_final.predict(X_test)
print(mean_absolute_error(y_test,forest_test_pred))
plt.scatter(X_test['sqft_living'],y_test,marker='*')
plt.scatter(X_test['sqft_living'],forest_test_pred,color='r')
plt.show()
constant=['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront',
       'condition', 'grade', 'sqft_basement','grade_cond','bed_bath_floors']
print('max error in test set',max(abs(forest_test_pred-y_test)))
print('minimum error in test set ',min(abs(forest_test_pred-y_test)))
print('mean error over test set ',np.average(abs(forest_test_pred-y_test)))
print("mean abs error over training set ",mean_absolute_error(y_train,forest_final.predict(X_train)))
forest_valid_pred=forest_final.predict(X_valid)
print('mean error over validation set',mean_absolute_error(y_valid,forest_valid_pred))
forest_final1=RandomForestRegressor(max_depth=20)
scores=cross_val_score(forest_final1,df.drop('price',axis=1),df['price'],cv=10,scoring='neg_mean_absolute_error')
print(scores)
print(np.average(scores))
#print(X_train.head(5))
#norms=norm(X_train[constant],axis=0)
#train=X_train[constant]/norms
#valid=X_valid[constant]/norms
#test=X_test[constant]/norms
#print(train.head(5))
#print(df['sqft_living'].head())
import numpy as np
#for c in np.logspace(4,10,5):
  #  for g in np.logspace(-4,8,12):
 #       reg=SVR(kernel='rbf',C=c,gamma=g)
   #     reg.fit(train[['sqft_living','bedrooms','bathrooms']],y_train)
    #    print('avg price of house in validation set',np.average(y_valid))
     #   y_train_pred=reg.predict(valid[['sqft_living','bedrooms','bathrooms']])
      #  print('abs error for C = ',c,"and gamma = ",g," is =",mean_absolute_error(y_valid,y_train_pred))
        #plt.scatter(valid['sqft_living'],y_valid)
        #plt.plot(valid['sqft_living'],y_train_pred,color='r')

