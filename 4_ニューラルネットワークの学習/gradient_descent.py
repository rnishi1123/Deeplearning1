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

def gradient_descent(f,init_x,lr=0.01,step_num=100):#lrは学習率learning rate　numは勾配法の繰り返し数
    x=init_x#初期値

    for i in range(step_num):
        grad = numerical_gradient(f,x)
        x -=lr*grad

    return x

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0,4.0])
print(gradient_descent(function_2,init_x=init_x,lr=0.1,step_num=100))
