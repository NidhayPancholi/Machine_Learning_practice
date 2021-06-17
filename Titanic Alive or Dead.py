import pandas as pd
import numpy as np
import seaborn as sns
train=pd.read_csv('titanic train.csv')
test=pd.read_csv('titanic test.csv')
print(train.head())
train.set_index('PassengerId',inplace=True)
test.set_index('PassengerId',inplace=True)
for x in train.columns:
    print(x)
    print(sum(train[x].isna()))
    print(len(train[x].unique()))
import matplotlib.pyplot as plt
train.drop(['Cabin','Name'],axis=1,inplace=True)
test.drop(['Cabin','Name'],axis=1,inplace=True)
train['Age'].replace(np.nan,np.median(train['Age'].dropna()),inplace=True)
train['Sex'].replace({'male':1,'female':0},inplace=True)
train.drop('Ticket',inplace=True,axis=1)
sns.distplot(a=train[train['Survived']==0]['Age'],label='dead')
sns.distplot(a=train[train['Survived']==1]['Age'],label='Survived')
plt.legend()
test['Age'].replace(np.nan,np.median(test['Age'].dropna()),inplace=True)
test['Sex'].replace({'male':1,'female':0},inplace=True)
test.drop('Ticket',inplace=True,axis=1)
#train.drop(['Parch','SibSp'],axis=1,inplace=True)
#test.drop(['Parch','SibSp'],axis=1,inplace=True)
train.dropna(inplace=True)
print(len(train))
embarked=pd.get_dummies(train['Embarked'])
embarked_test=pd.get_dummies(test['Embarked'])
train=train.join(embarked)
test=test.join(embarked_test)
train.drop("Embarked",axis=1,inplace=True)
test.drop("Embarked",axis=1,inplace=True)

test['Fare'].replace(np.nan,np.mean(test['Fare'].dropna()),inplace=True)
test.dropna(inplace=True)
pclass=pd.get_dummies(train['Pclass'])
pclass_test=pd.get_dummies(test['Pclass'])
train.drop('Fare',axis=1,inplace=True)
test.drop('Fare',axis=1,inplace=True)
#train['Pclass'].replace({3:1,2:2,1:3},inplace=True)
#test['Pclass'].replace({3:1,2:2,1:3},inplace=True)
#train=train.join(pclass)
#test=test.join(pclass_test)
#train.drop('Pclass',axis=1,inplace=True)
#test.drop('Pclass',axis=1,inplace=True)
print(train.columns)
print(test.columns)
X_train,y_train=train.drop('Survived',axis=1),train['Survived']
test=test[X_train.columns]
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(
 RandomForestClassifier(max_depth=3,n_estimators=100), n_estimators=10,
 algorithm="SAMME.R", learning_rate=0.75
 )
ada_clf.fit(X_train, y_train)
clf=RandomForestClassifier(n_estimators=10000,max_depth=2)
#clf.fit(X_train,y_train)
y_pred=ada_clf.predict(test)
#print(clf.feature_importances_)
#print(y_pred)
final_df=pd.DataFrame({'PassengerId':test.index,'Survived':y_pred})
print(final_df)
final_df.set_index('PassengerId',inplace=True)
final_df.to_csv(r'C:\Users\Nidhay Pancholi\PycharmProjects\The 20HourProject\final_file27.csv')
