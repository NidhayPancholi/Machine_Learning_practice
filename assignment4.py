def degree(df,feature,maximal_degree):
    new_df=df[feature]
    for x in feature:
        for y in range(2,maximal_degree+1):
            new_df[x+str(y)]=df[x]**y
    #new_df['price']=df['price']
    return  new_df
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
from sklearn.model_selection import KFold
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort_values(['sqft_living','price'])
sales_15=degree(sales,['sqft_living'],15)
reg=Ridge(alpha=1.5e-5,normalize=True)
reg.fit(sales_15,sales['price'])
print(sales_15.columns)
print(reg.coef_)

print("______________________________________________________________")
set_1 = pd.read_csv('wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
reg_set1=Ridge(alpha=1.23e2,normalize=True)
set1=degree(set_1,['sqft_living'],15)
reg_set1.fit(set1,set_1['price'])
print(reg_set1.coef_)
reg_set2=Ridge(alpha=1.23e2,normalize=True)
set2=degree(set_2,['sqft_living'],15)
reg_set2.fit(set2,set_2['price'])
print(reg_set2.coef_)
reg_set3=Ridge(alpha=1.23e2,normalize=True)
set3=degree(set_3,['sqft_living'],15)
reg_set3.fit(set3,set_3['price'])
print(reg_set3.coef_)
reg_set4=Ridge(alpha=1.23e2,normalize=True)
set4=degree(set_4,['sqft_living'],15)
reg_set4.fit(set4,set_4['price'])
print(reg_set4.coef_)

print("_________________________________________________")

train= pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
kf=KFold(n_splits=10)
y=train['price']
n = len(train)
h=[]
k = 10 # 10-fold cross-validation
for y in np.logspace(3, 9, num=13):
    sum=0
    for i in range(k):
        start = (n * i) // k
        end = (n * (i + 1)) // k
        X_val=train[start:end+1]
        X_train=train[0:start].append(train[end+1:n])
        x_train=degree(X_train,['sqft_living'],15)
        x_val=degree(X_val,['sqft_living'],15)
        y_train=X_train['price']
        y_val=X_val['price']
        ridge=Ridge(alpha=y,normalize=True)
        ridge.fit(x_train,y_train)
        y_pred=ridge.predict(x_val)
        sum+=(mean_squared_error(y_val,y_pred)*len(y_val))
    h.append(sum/10)

print(h)
print(min(h))
print(h.index(min(h)))
print(np.logspace(3,9,num=13)[h.index(min(h))])
train= pd.read_csv('wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
final_train=degree(train,['sqft_living'],15)
final_test=degree(test,['sqft_living'],15)
r1=Ridge(alpha=1000,normalize=True)
r1.fit(final_train,train['price'])
y_test=test['price']
y_test_pred=r1.predict(final_test)
print(mean_squared_error(y_test,y_test_pred)*len(y_test))