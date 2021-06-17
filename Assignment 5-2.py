import pandas as pd
import numpy as np
import  math
def func(dataframe,features,output):
    train=pd.DataFrame()
    train['con']=[1 for x in range(len(dataframe))]
    for x in features:
        train[x]=dataframe[x]
    output_array=dataframe[output]
    train['con']=normalize(train[['con','sqft_living','bedrooms']],norm='l1')
    return np.array(train),np.array(output_array)
def dot(features,weights):
    return np.dot(features,weights)
training=pd.read_csv('kc_house_train_data.csv')
#training=pd.read_csv('kc_house_train_data.csv')
testing=pd.read_csv("kc_house_test_data.csv")
#testing=pd.read_csv('kc')
from sklearn.preprocessing import normalize
def norm(features):
    norms=np.linalg.norm(features,axis=0)
    return features/norms
#def lasso(features,output,weight,tolerance):
 #   new_weights=weight.copy()
  #  prev_weights=np.zeros(weight.shape)
   # while True:
    #    for x in range(len(weight)):
     #       prediction=dot(features,new_weights)
      #      ro_j=np.sum((features[:,x]*(output-prediction+new_weights[x]*features[:,x])))

       # change=new_weights-prev_weights
        #if max(change)<tolerance:
         #   return new_weights
        #prev_weights=new_weights


def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    new_weights = initial_weights.copy()  # instantiate new_weights list to be replaced as loops through coordinates/features
    coord_changes = [np.NaN for i in range(len(initial_weights))]

    should_continue = True  # flag to indicate whether to continue looping through while loop or to break and return new_weights
    prev_weights=np.zeros(len(initial_weights))
    while should_continue:

        for i in range(feature_matrix.shape[1]):
            new_weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, new_weights, l1_penalty)
            #coord_changes[i] = np.abs(prev_weights[i] - new_weights[i]) # change in coordinates (absolute value)
        change=np.array(new_weights)-np.array(prev_weights)
        if max(change) < tolerance:
            should_continue = False
        else :
            prev_weights=new_weights
    return new_weights


def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    # compute prediction
    prediction = dot(feature_matrix, weights)

    # compute ro[i] = SUM[ [feature_i]*(output - prediction + weight[i]*[feature_i]) ]
    ro_i = np.sum(feature_matrix[:, i] * (output - prediction + weights[i] * feature_matrix[:, i]))
    print(i,'      ',ro_i)
    if i == 0:  # intercept -- do not regularize
        new_weight_i = ro_i
    elif ro_i < -l1_penalty / 2.:
        new_weight_i = ro_i + (l1_penalty / 2.)
    elif ro_i > l1_penalty / 2.:
        new_weight_i = ro_i - (l1_penalty / 2.)
    else:
        new_weight_i = 0.

    return new_weight_i



feature,output=func(training,['sqft_living','bedrooms'],'price')
norms=np.linalg.norm(feature,axis=0)
feature=feature/norms

coef=lasso_cyclical_coordinate_descent(feature,output,[0,0,0],1e7,1.0)
print(coef)
pred=dot(feature,coef)
print(sum((pred-output)**2))
l=['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated']
feature2,output2=func(training,l,'price')
norms2=np.linalg.norm(feature2,axis=0)
feature2=feature2/norms2
coef1e7=lasso_cyclical_coordinate_descent(feature2,output2,np.zeros(len(feature2[0])),1e7,1.0)
coef1e4=lasso_cyclical_coordinate_descent(feature2,output2,np.zeros(len(feature2[0])),1e4,5e5)
coef1e8=lasso_cyclical_coordinate_descent(feature2,output2,np.zeros(len(feature2[0])),1e8,1.0)
print(coef1e4)
print(l)
print(coef1e7)
for x in range(len(coef1e7)):
    if coef1e7[x]==0:
        print(l[x-1])

test_feature,test_output=func(testing,l,'price')
test_feature=test_feature/norms2
pred1e7=dot(test_feature,coef1e7)
pred1e8=dot(test_feature,coef1e8)
pred1e4=dot(test_feature,coef1e4)
print(sum((test_output-pred1e4)**2))
print(sum((test_output-pred1e7)**2))
print(sum((test_output-pred1e8)**2))
