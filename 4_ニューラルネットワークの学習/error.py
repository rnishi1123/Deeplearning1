import numpy as np

def mean_squared_error(y,t):#誤差が小さい方が良い
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    delta = 1e-7#log(0)を防ぐための微細な値
    return -np.sum(t*np.log(y+delta))

t=[0,0,1,0,0,0,0,0,0,0]

y=[0.1,0.05,0.6,0,0.05,0.1,0,0.1,0,0]
print("mean=",mean_squared_error(np.array(y),np.array(t)))
print("cross=",cross_entropy_error(np.array(y),np.array(t)))

y=[0.1,0.05,0.1,0,0.05,0.1,0,0.6,0.5,0]
print("mean=",mean_squared_error(np.array(y),np.array(t)))
print("cross=",cross_entropy_error(np.array(y),np.array(t)))