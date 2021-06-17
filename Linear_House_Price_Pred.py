import pandas as pd
df=pd.read_csv('kc_house_train_data.csv')
df=df.astype({'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
print(df.dtypes)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
l=LinearRegression()
l.fit(df[['sqft_living']],df['price'])
print(l.coef_)
print(l.intercept_)
print(l.predict([[2650]]))
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
y_pred=l.predict(df[['sqft_living']])
print(mean_squared_error(df['price'],y_pred)*len(df['price']))
l1=LinearRegression()
l1.fit(df[['bedrooms']],df['price'])
test_df=pd.read_csv("kc_house_test_data.csv")
test_df=test_df.astype({'bathrooms':float, 'waterfront':int,'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int})
y_pred2=l1.predict(test_df[['bedrooms']])
y_predict=l.predict(test_df[['sqft_living']])
print("Bedrooms RSS Score=",mean_absolute_error(test_df['price'],y_pred2)*len(test_df['price']))
print("Sqft_living RSS Score=",mean_absolute_error(test_df['price'],y_predict)*len(test_df['price']))
print("_________________________________________________________________________________________")
