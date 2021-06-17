import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('tmdb_5000_movies.csv')
print(df.head(5))
g=df.drop(['genres','production_companies','production_countries','spoken_languages','homepage'],axis=1)
print(df.columns)
g['points']=g['vote_average']*g['vote_count']
g['rank']=g['points'].rank(ascending=False)
h=g.sort_values('rank',ascending=True)
h['budget_in_billions']=h['budget']/100000000
f=h[h['rank']<=100]
#f.plot.line(x='rank',y='budget_in_billions')
#f.plot.line(x='rank',y='runtime')
#ni=f.sort_values('runtime',ascending=False)
#ni.plot.line(y='budget_in_billions',x='runtime')
f['profit_in_billions']=(f['revenue']-f['budget'])/1000000000
print(f[['title','revenue','budget','profit_in_billions']])
f['revenue_in_billions']=f['revenue']/100000000
f=f.drop(['budget','revenue'],axis=1)
print(f.columns)
f.set_index('rank')
f.sort_index()
print(f[['title','runtime','profit_in_billions']])
f['runtime_in_hrs']=f['runtime']/60
nid=f[['runtime_in_hrs','profit_in_billions','budget_in_billions']]
#f.plot.scatter(x='profit_in_billions',y='runtime')
plt.hist2d(x=nid['runtime_in_hrs'],y=nid['profit_in_billions'])
plt.colorbar()
plt.xlabel('Runtime in hours')
plt.ylabel('Profit the Movie Earnt in Billions')
plt.show()
heatmap=plt.hist2d(x=nid['budget_in_billions'],y=nid['profit_in_billions'])
plt.colorbar()
plt.xlabel('Budget in Billions')
plt.ylabel('Profit in Billions')
plt.show()