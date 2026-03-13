import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f,x):
    h = 1e-4 #0.0001
    return (f(x+h)-f(x-h))/(2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x

x=np.arange(0,20,0.1)
y=function_1(x)

z=numerical_diff(y,5)
a=numerical_diff(y,10)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x,y,z,a)
plt.show()
