import pandas as pd
df=pd.read_csv('datasets_244521_516703_nifty_500_stats.csv',delimiter=';')
df['Dividend']=df['current_value']*df['dividend_yield']/100
df.drop(['symbol','dividend_yield'],axis=1,inplace=True)
cate=pd.get_dummies(df['category'])
df=df.join(cate)
df.drop('category',axis=1,inplace=True)
industry=pd.get_dummies(df['industry'])
df=df.join(industry)
df.drop(['Unnamed: 0','industry'],axis=1,inplace=True)
df.drop('price_earnings',axis=1,inplace=True)
df.dropna(axis=0,inplace=True)
from sklearn.preprocessing import MinMaxScaler
m=MinMaxScaler()
print(df.dtypes)
print(df['market_cap'])
df[['market_cap','current_value','high_52week','low_52week','book_value','sales_growth_3yr']]=m.fit_transform(df[['market_cap','current_value','high_52week','low_52week','book_value','sales_growth_3yr']])
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df.drop('Dividend',axis=1),df['Dividend'],random_state=0)
print(X_train.dtypes)
train_companies=X_train['company']
test_companies=X_test['company']
X_train.drop('company',axis=1,inplace=True)
X_test.drop('company',axis=1,inplace=True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
for x in range(2,15):
    for y in np.linspace(100,1001,5):
        reg=RandomForestRegressor(max_depth=x,n_estimators=int(y))
        reg.fit(X_train,y_train)
        y_pred=reg.predict(X_test)
        print('max_deth = ',x,' n_estimators = ',int(y),' mean_abs_error = ',mean_absolute_error(y_test,y_pred),' mean_sq_error = ',mean_squared_error(y_test,y_pred),' max error = ',max(abs(y_pred-y_test)),' min error = ',min(abs(y_pred-y_test)))
print(np.mean(y_test))
reg=RandomForestRegressor(max_depth=11,n_estimators=100)
reg.fit(X_train,y_train)
y_pred=reg.predict(X_test)
final_df=pd.DataFrame({'company':test_companies,'pred_value':y_pred,'true_value':y_test})
final_df.to_csv(r'C:\Users\Nidhay Pancholi\PycharmProjects\The 20HourProject\stock2.csv')