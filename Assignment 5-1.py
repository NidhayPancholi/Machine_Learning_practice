import pandas as pd
import numpy as np
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('kc_house_data.csv', dtype=dtype_dict)

all_features = ['bedrooms', 'bedrooms_square',
            'bathrooms',
            'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt',
            'floors', 'floors_square',
            'waterfront', 'view', 'condition', 'grade',
            'sqft_above',
            'sqft_basement',
            'yr_built', 'yr_renovated']

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']
from sklearn.linear_model import Lasso
all_feature=Lasso(alpha=5e2,normalize=True)
all_feature.fit(sales[all_features],sales['price'])
for x in range(len(all_feature.coef_)):
    if all_feature.coef_[x]!=0:
        print(all_features[x])


testing = pd.read_csv('wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('wk3_kc_house_valid_data.csv', dtype=dtype_dict)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']
h=[]
from sklearn.metrics import mean_squared_error
for x in np.logspace(1,7,13):
    lasso=Lasso(alpha=x,normalize=True)
    lasso.fit(training[all_features],training['price'])
    y_pred=lasso.predict(validation[all_features])
    h.append(mean_squared_error(validation['price'],y_pred)*len(y_pred))
    if x==10.0:
        count=0
        for y in range(len(lasso.coef_)):
            if y!=0:
                print(all_features[y])
                count+=1
        print(count)

laso=Lasso(alpha=10,normalize=True)
laso.fit(training[all_features],training['price'])
print(laso.coef_)
print(laso.intercept_)
print(h)
print(min(h))
print(np.logspace(1,7,13)[h.index(min(h))])
h=[]
print("________________________________________")
for x in np.logspace(1,4,20):
    model=Lasso(alpha=x,normalize=True)
    model.fit(training[all_features],training['price'])
    #h.append((len(all_features)-list(model.coef_).count(0)))
    h.append(np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_))
print(h)
print(np.logspace(1,4,20)[h.index(7)])
print(np.logspace(1,4,20))
print("_________________________________")
l1_penalty_max=264#.66508987
l1_penalty_min=127#.42749857
u=[]
h=[]
for x in np.linspace(l1_penalty_min,l1_penalty_max,20):
    lasso1=Lasso(alpha=x,normalize=True)
    lasso1.fit(training[all_features],training['price'])
    if np.count_nonzero(lasso1.coef_) + np.count_nonzero(lasso1.intercept_)==7:
        y_pred1=lasso1.predict(validation[all_features])
        u.append(mean_squared_error(validation['price'],y_pred1)*len(y_pred1))
        h.append(x)
print(h[u.index(min(u))])
print(min(u))
print(u)
l=155.8421052631579
lasso2=Lasso(alpha=l,normalize=True)
lasso2.fit(training[all_features],training['price'])
for x in range(len(lasso2.coef_)):
    if lasso2.coef_[x]!=0:
        print(all_features[x])
