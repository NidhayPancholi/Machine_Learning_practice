import pandas as pd
df=pd.DataFrame(['04/20/2009', '04/20/09', '4/20/09', '4/3/09', 'Mar-20-2009', 'Mar 20, 2009', 'March 20, 2009', 'Mar 20 2009',' 12/2009'])
df=pd.to_datetime(df[0],infer_datetime_format='dd/mm/yy')
print(df)
print(pd.to_datetime('Mar-20-2009'))