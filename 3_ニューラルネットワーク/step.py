import numpy as np
import matplotlib.pylab as plt

def step_function(x):
    y=x>0
    return y.astype(np.int64)

x=np.arange(-5,5,0.1)
y=step_function(x)
plt.plot(x,y)
plt.ylim(-0.1,1.1)
plt.show()