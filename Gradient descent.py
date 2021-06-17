import pandas as pd
import numpy as np
def func(dataframe,features,output):
    train=pd.DataFrame()
    train['con']=[1 for x in range(len(dataframe))]
    for x in features:
        train[x]=dataframe[x]
    output_array=dataframe[output]
    return np.array(train),np.array(output_array)
def y_hat(features,weights):
    return np.dot(features,weights)
def errors(y_true,y_pred):
    return y_true-y_pred
def derivative(error,feature):
    return -2*np.dot(np.transpose(feature),error)
def descent(features,output,weights,step_size,tolerance):
    while True:
        rss=0
        for x in range(len(weights)):
            y_pred = y_hat(features, weights)
            error = errors(np.array(output), y_pred)
            weights[x]-=(step_size*derivative(error,features[:,x]))
            sum=0
            for x in error:
                sum+=(x**2)
            rss+=sum
        print(rss)
        print(weights)
        if np.sqrt(rss)<tolerance:
            return weights
df=pd.read_csv('kc_house_train_data.csv')
feature,output=func(df,['sqft_living'],'price')
w=descent(feature,output,[-47000.0,1.0],7e-12,2.5e7)
test_df=pd.read_csv('kc_house_test_data.csv')
test_df['pred']=w[0]+w[1]*test_df['sqft_living']
print(w)
