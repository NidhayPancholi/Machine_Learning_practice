import pandas as pd
import numpy as np
df=pd.read_csv('kc_house_test_data.csv')
def func(dataframe,features,output):
    train=pd.DataFrame()
    train['con']=[1 for x in range(len(dataframe))]
    for x in features:
        train[x]=dataframe[x]
    output_array=dataframe[output]
    return np.array(train),np.array(output_array)
def dot(features,weights):
    return np.dot(features,weights)
def derivative(features,errors):
    #features=np.transpose(features)
    return 2*np.dot(features,errors)
def descent(features,output,weights,step_size,tolerance):
    weight=np.array(weights)
    output=np.array(output)
    while True:
        errors = dot(features, weight)-output
        grad_mag=0
        for x in range(len(weight)):
            #print(errors)
            grad_mag += (derivative(features[:, x], errors) ** 2)
            weight[x]=weight[x]-step_size*derivative(features[:,x],errors)
        #print(grad_mag)

        if np.sqrt(grad_mag)<tolerance:
            return weight

feature,output=func(df,['sqft_living'],'price')
weights=descent(feature,output,[-47000.0,1.0],7e-12,2.5e7)
print(weights[0])
print(weights[1])
test_df=pd.read_csv('kc_house_test_data.csv')
print(test_df.iloc[0])
test_df['pred_value']=weights[1]*test_df['sqft_living']+weights[0]
print(test_df['pred_value'][0])
feature2,output2=func(df,['sqft_living','sqft_living15'],'price')
weights2=descent(feature2,output2,[-100000,1,1],4e-12,1e9)
print(weights2)

test_df['pred_value']=weights2[1]*test_df['sqft_living']+weights2[0]+weights2[2]*test_df['sqft_living15']
print(test_df['pred_value'][0])