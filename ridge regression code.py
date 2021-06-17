import pandas as pd
import numpy as np
df=pd.read_csv('kc_house_train_data.csv')
def func(dataframe,features,output):
    train=pd.DataFrame()
    train['con']=[1 for x in range(len(dataframe))]
    for x in features:
        train[x]=dataframe[x]
    output_array=dataframe[output]
    return np.array(train),np.array(output_array)
def dot(features,weights):
    return np.dot(features,weights,)
def derivative(features,errors,l2_pen,w):
    #features=np.transpose(features)
    return 2*np.dot(features,errors)+2*w*l2_pen
def descent(features,output,weights,step_size,max_iter,l2_pen):
    weight=np.array(weights)
    output=np.array(output)
    iter=0
    while True:
        errors = dot(features, weight)-output
        grad_mag=0
        for x in range(len(weight)):
            if x==0:
                grad_mag += (derivative(features[:, x], errors,0,weight[x]) ** 2)
                weight[x] = weight[x] - step_size * derivative(features[:, x], errors,0,weight[x])
            #print(errors)
            else:
                grad_mag += (derivative(features[:, x], errors,l2_pen,weight[x]) ** 2)
                weight[x]=weight[x]-step_size*derivative(features[:,x],errors,l2_pen,weight[x])
        #print(grad_mag)
        iter+=1
        if iter>=max_iter:
            return weight
feature,output=func(df,['sqft_living','sqft_living15'],'price')
no_pen_coeff=descent(feature,output,[0.0,0.0,0.0],1e-12,1000,0)
print(no_pen_coeff)
print(round(no_pen_coeff[1],1))
test_df=pd.read_csv('kc_house_test_data.csv')[['sqft_living','price','sqft_living15']]
test_df['pred']=no_pen_coeff[1]*test_df['sqft_living']+no_pen_coeff[0]+no_pen_coeff[2]*test_df['sqft_living15']
test_df['error']=test_df['pred']-test_df['price']
print(sum(test_df['error']**2))
high_pen=descent(feature,output,[0.0,0.0,0.0],1e-12,1000,1e11)
test_df['pred2']=high_pen[1]*test_df['sqft_living']+high_pen[0]+high_pen[2]*test_df['sqft_living15']
print(test_df['price'][0]-test_df['pred'][0])

print(test_df['price'][0]-test_df['pred2'][0])