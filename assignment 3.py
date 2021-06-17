import pandas as pd
df=pd.read_csv('wk3_kc_house_train_data.csv')
df=df[['sqft_living','price']]
df=df.sort_values(['sqft_living','price'])
def degree(df,feature,maximal_degree):
    new_df=df[feature]
    for x in feature:
        for y in range(2,maximal_degree+1):
            new_df[x+str(y)]=df[x]**y
    new_df['price']=df['price']
    return  new_df
train_df=degree(df,['sqft_living'],3)
X_train=train_df.drop(['price'],axis=1)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,df['price'])
import matplotlib.pyplot as plt
plt.plot(train_df['sqft_living'],train_df['price'],'.')
plt.plot(train_df['sqft_living'],reg.predict(X_train),'-')
#plt.show()
subset1=pd.read_csv('wk3_kc_house_set_1_data.csv')[['sqft_living','price']]
subset2=pd.read_csv('wk3_kc_house_set_2_data.csv')[['sqft_living','price']]
subset3=pd.read_csv('wk3_kc_house_set_3_data.csv')[['sqft_living','price']]
subset4=pd.read_csv('wk3_kc_house_set_4_data.csv')[['sqft_living','price']]
subset1=degree(subset1,['sqft_living'],15)
print(subset1.columns)
reg1=LinearRegression()
reg1.fit(subset1.drop(['price'],axis=1),subset1['price'])
plt.plot(subset1['sqft_living'],subset1['price'],'.')
plt.plot(subset1['sqft_living'],reg1.predict(subset1.drop(['price'],axis=1)),'-')
#plt.show()
subset2=degree(subset2,['sqft_living'],15)
reg2=LinearRegression()
reg2.fit(subset2.drop(['price'],axis=1),subset2['price'])
plt.plot(subset2['sqft_living'],subset2['price'],'.')
plt.plot(subset2['sqft_living'],reg2.predict(subset2.drop(['price'],axis=1)),'-')
#plt.show()
subset3=degree(subset3,['sqft_living'],15)
reg3=LinearRegression()
reg3.fit(subset3.drop(['price'],axis=1),subset3['price'])

subset4=degree(subset4,['sqft_living'],15)
reg4=LinearRegression()
reg4.fit(subset4.drop(['price'],axis=1),subset4['price'])
#print('1',reg1.coef_)
#print('2',reg2.coef_)
#print('3',reg3.coef_)
#print('4',reg4.coef_)
val_df=pd.read_csv('wk3_kc_house_valid_data.csv')[['sqft_living','price']]
test_df=pd.read_csv('kc_house_test_data.csv')[['sqft_living','price']]
y_val=val_df['price']
y_train=train_df['price']
h=[]
from sklearn.metrics import mean_squared_error
for x in range(1,16):
    X_train=degree(train_df,['sqft_living'],x)
    val_df=degree(val_df,['sqft_living'],x)
    l=LinearRegression()
    l.fit(X_train.drop(['price'],axis=1),y_train)
    y_pred=l.predict(val_df.drop(['price'],axis=1))
    h.append(mean_squared_error(y_val,y_pred)*len(y_val))
print(h.index(min(h)))
print(min(h))
print(h)
y_test=test_df['price']
X_train=degree(train_df,['sqft_living'],15)
X_test=degree(test_df,['sqft_living'],15).drop(['price'],axis=1)
l_test=LinearRegression()
l_test.fit(X_train.drop(['price'],axis=1),y_train)
y_test_pred=l_test.predict(X_test)
print(mean_squared_error(y_test,y_test_pred)*len(y_test))
