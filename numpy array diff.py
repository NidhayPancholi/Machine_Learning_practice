import numpy as np
a=np.array([[1,2,3],[4,5,6]])
b=np.array([1,5,0])
print(a-b)
print(np.sum((a-b)**2,axis=1))
print(np.sqrt(np.sum((a-b)**2,axis=1)))