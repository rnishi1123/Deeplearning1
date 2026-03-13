import numpy as np

def numerical_gradient(f,x):
    h=1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val=x[idx]
        #f(x+h)の計算
        x[idx]=tmp_val+h
        fxh1 = f(x)

        #f(x-h)の計算
        x[idx]=tmp_val-h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2)/(2*h)
        x[idx] = tmp_val#値を元に戻す
    
    return grad

def numerical_diff(f, x):
    h = 1e-4 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01*x**2 + 0.1*x 

def function_2(x):
    return np.sum(x**2)

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

print(numerical_gradient(function_2,np.array([3.0,4.0])))

print(numerical_gradient(function_2,np.array([0.0,2.0])))

print(numerical_gradient(function_2,np.array([3.0,0.0])))

