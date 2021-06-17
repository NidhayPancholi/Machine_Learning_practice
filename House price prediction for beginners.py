import pandas as pd
import numpy as np
train=pd.read_csv('train housing.csv',index_col='Id')
#pd.set_option('display.max_columns',None)
test=pd.read_csv('test housing.csv',index_col='Id')
#train['HowOld']=2010-train['YearBuilt']
#test['HowOld']=2010-test['YearBuilt']
train.drop(['MSZoning','LotFrontage','Alley','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Condition1','Condition2','BldgType'
 ,'YearBuilt','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','Foundation','BsmtQual'
,'BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','Heating','Electrical','LowQualFinSF',
'GrLivArea','Functional','FireplaceQu','GarageYrBlt','GarageFinish','GarageCars','GarageQual','PavedDrive','3SsnPorch','ScreenPorch','PoolQC','Fence'
,'MiscFeature','MoSold','YrSold','SaleType','MSSubClass','YearRemodAdd'],axis=1,inplace=True)
test.drop(['MSZoning','LotFrontage','Alley','Street','LotShape','LandContour','Utilities','LotConfig','LandSlope','Condition1','Condition2','BldgType'
 ,'YearBuilt','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','Foundation','BsmtQual'
,'BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','Heating','Electrical','LowQualFinSF',
'GrLivArea','Functional','FireplaceQu','GarageYrBlt','GarageFinish','GarageCars','GarageQual','PavedDrive','3SsnPorch','ScreenPorch','PoolQC','Fence'
,'MiscFeature','MoSold','YrSold','SaleType','MSSubClass','YearRemodAdd'],axis=1,inplace=True)

test['GarageType'].replace({np.nan:0},inplace=True)
train['GarageType'].replace({np.nan:0},inplace=True)
neighbor=pd.get_dummies(train['Neighborhood'])
tneighbor=pd.get_dummies(test['Neighborhood'])
house_style=pd.get_dummies(train['HouseStyle'])
thouse_style=pd.get_dummies(test['HouseStyle'])
train['ExterCond'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10},inplace=True)
test['ExterCond'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10},inplace=True)
train['BsmtCond'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10,np.nan:0},inplace=True)
test['BsmtCond'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10,np.nan:0},inplace=True)
train['HeatingQC'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10},inplace=True)
test['HeatingQC'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10},inplace=True)
train['CentralAir'].replace({'Y':1,'N':0},inplace=True)
test['CentralAir'].replace({'Y':1,'N':0},inplace=True)
train['Baths']=train['BsmtFullBath']+train['BsmtHalfBath']*0.5+train['HalfBath']*0.5+train['FullBath']
test['Baths']=test['BsmtFullBath']+test['BsmtHalfBath']*0.5+test['HalfBath']*0.5+test['FullBath']
train.drop(['BsmtFullBath','BsmtHalfBath','HalfBath','FullBath'],axis=1,inplace=True)
test.drop(['BsmtFullBath','BsmtHalfBath','HalfBath','FullBath'],axis=1,inplace=True)
train['KitchenQual'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10,np.nan:0},inplace=True)
test['KitchenQual'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10,np.nan:0},inplace=True)
train['GarageCond'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10,np.nan:0},inplace=True)
test['GarageCond'].replace({'TA':5,'Gd':8,'Fa':3,'Po':1,'Ex':10,np.nan:0},inplace=True)
sale_cond=pd.get_dummies(train['SaleCondition'])
tsale_cond=pd.get_dummies(test['SaleCondition'])
train=train.join(neighbor)
test=test.join(tneighbor)
train=train.join(house_style)
test=test.join(thouse_style)
train.drop(['Neighborhood','HouseStyle','GarageType','SaleCondition','OpenPorchSF','EnclosedPorch','WoodDeckSF','PoolArea','MiscVal',
            'ExterCond','BsmtCond','HeatingQC','CentralAir'],axis=1,inplace=True)
test.drop(['Neighborhood','HouseStyle','GarageType','SaleCondition','OpenPorchSF','EnclosedPorch','WoodDeckSF','PoolArea','MiscVal',
           'ExterCond','BsmtCond','HeatingQC','CentralAir'],axis=1,inplace=True)
test['TotalBsmtSF'].replace({np.nan:0},inplace=True)
test['GarageArea'].replace({np.nan:0},inplace=True)
test['Baths'].replace({np.nan:0},inplace=True)


X_train=train.drop('SalePrice',axis=1)
X_train=X_train[X_train['LotArea']<60000]
y_train=train['SalePrice']
print(train.columns)
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
forest=RandomForestRegressor(max_depth=15,n_estimators=1000)
forest.fit(X_train,y_train/1000000)
test['TotalBsmtSF'].replace({np.nan:0},inplace=True)
test['GarageArea'].replace({np.nan:0},inplace=True)
test['Baths'].replace({np.nan:0},inplace=True)
test['2.5Fin']=0
test=test[X_train.columns]
ada=AdaBoostRegressor(RandomForestRegressor(max_depth=10),n_estimators=100,learning_rate=0.5)
ada.fit(X_train,y_train/1000000)
y_pred=ada.predict(test)
final_df=pd.DataFrame({'Id':test.index,'SalePrice':y_pred*1000000})
final_df.set_index('Id',inplace=True)
final_df.to_csv(r'C:\Users\Nidhay Pancholi\PycharmProjects\The 20HourProject\house_abs_error_kaggle1.csv')
print(list(zip(X_train.columns,forest.feature_importances_)))