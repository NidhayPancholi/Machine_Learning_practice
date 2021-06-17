import matplotlib.pyplot as plt
import pandas as pd
freq=[1,2,5,10,50,100,250,500,750,1000,1250,
1500,0.5,0.1,0.01,0.001,0.0001,0.00001,
0.0005,2000,10000,15000]
gain=[0.864067865,0.85691394,0.856883895,0.856943985,0.860110566,0.85678891,0.858131038,0.857322344,0.856914245,0.860139,
0.857534634,0.852769270,0.856557951,0.855907262,0.856293002,0.856282987,0.85445022,0.60168855,0.838866689,0.853526311,
0.886272039,0.85]
df=pd.DataFrame({"Frequency":freq,'Gain':gain})
df.sort_values(by='Frequency',inplace=True)
plt.plot(df['Frequency'],df['Gain'])
plt.show()


freq2=[1,10,25,50,100,250,500,750,1000,2000,5000,10000,15000,20000,30000,0.5,0.1,0.001,50000,60000,75000]
gain2=[6.3673,6.3618,6.3624,6.3619,6.362,6.3618,6.36,6.359,6.359,6.365,6.356,6.341,6.356,6.40,6.336,6.36,6.35,6.33,6.24,6.16,6.17]
df=pd.DataFrame({"Frequency":freq2,'Gain':gain2})
df.sort_values(by='Frequency',inplace=True)
plt.plot(df['Frequency'],df['Gain'])
plt.show()
len(gain2)