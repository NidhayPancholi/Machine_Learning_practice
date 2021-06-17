import pandas as pd
def degree(df,feature,maximal_degree):
    new_df=df[feature]
    for x in feature:
        for y in range(2,maximal_degree+1):
            new_df[x+str(y)]=df[x]**y
    #new_df['price']=df['price']
    return  new_df
X_train=pd.read_csv('wk3_kc_house_train_data.csv')[['sqft_living']]
y_train=pd.read_csv('wk3_kc_house_train_data.csv')['price']
X_val=pd.read_csv('wk3_kc_house_valid_data.csv')[['sqft_living']]
y_val=pd.read_csv('wk3_kc_house_valid_data.csv')['price']
X_test=pd.read_csv('wk3_kc_house_test_data.csv')[['sqft_living']]
y_test=pd.read_csv('wk3_kc_house_test_data.csv')['price']
h=[]
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
for x in range(1,16):
    X_train=degree(X_train,['sqft_living'],x)
    X_val=degree(X_val,['sqft_living'],x)
    l=LinearRegression()
    l.fit(X_train,y_train)
    y_pred=l.predict(X_val)
    error=y_val-y_pred
    h.append(sum(error**2))
print(min(h))
print(h)
print(h.index(min(h)))
l1=LinearRegression()
X_train=degree(X_train,['sqft_living'],6)
X_test=degree(X_test,['sqft_living'],6)
l1.fit(X_train,y_train)
y_test_pred=l1.predict(X_test)
print(mean_squared_error(y_test,y_test_pred)*len(y_test))


