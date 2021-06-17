import pandas as pd
df=pd.read_csv('datasets_244521_516703_nifty_500_stats.csv',delimiter=';')
l=(df['dividend_yield']>0).astype(int)
df['dividend_yield']=l
cate=pd.get_dummies(df['category'])
df=df.join(cate)
df.drop('category',axis=1,inplace=True)
industry=pd.get_dummies(df['industry'])
df=df.join(industry)
df.drop(['Unnamed: 0','industry'],axis=1,inplace=True)
df.drop('price_earnings',axis=1,inplace=True)
df.dropna(axis=0,inplace=True)
df.drop('symbol',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop('dividend_yield',axis=1),df['dividend_yield'],random_state=0)
print(X_train.dtypes)
train_companies=X_train['company']
test_companies=X_test['company']
X_train.drop('company',axis=1,inplace=True)
X_test.drop('company',axis=1,inplace=True)
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score,precision_recall_curve,precision_score
#for x in range(2,15):
 #   for y in np.linspace(100,1001,5):
  #      reg=RandomForestClassifier(max_depth=x,n_estimators=int(y))
   #     reg.fit(X_train,y_train)
    #    y_pred=reg.predict_proba(X_test)
     #   y_pred=(y_pred[:,1]>0.70).astype(int)
      #  print('max_deth = ', x, ' n_estimators = ', int(y),' accuracy = ',accuracy_score(y_test,y_pred),' precision = ',precision_score(y_test,y_pred))
reg=RandomForestClassifier(n_estimators=100,max_depth=8)
reg.fit(X_train,y_train)
y_pred=reg.predict_proba(X_test)
y_pred=(y_pred[:,1]>0.7).astype(int)
pre,rec,r=precision_recall_curve(y_test,y_pred)
import matplotlib.pyplot as plt
plt.plot(rec,pre)
plt.show()